// Upstream calls and routing.
//
// For each request the candidates are the free models that meet its
// requirements, ranked by the sealed quality table's rating, descending.
// A JSON-mode request (the client asked for it on the OpenAI wire) can
// only go to a wire that can carry it: OpenAI's response_format or
// Gemini's responseMimeType. The Anthropic wire has no JSON mode, so an
// anthropic-wire provider would echo it absent and the request would fail
// its echo check.
//
// Candidates are tried in rank order; a refusal (402/429 budget, a
// failure-schedule call, an outage, or any other non-2xx) moves on to the
// next candidate. A 2xx is accepted as the answer: retrying after a
// success would put two successful provider calls behind one request, and
// the trial counts that against it. Calls are awaited one at a time, so a
// provider never sees a second call while its first is in flight.
//
// Ties in rating break toward the provider whose wire matches the
// client's (less translation), then by provider id, then model id, so the
// order is a pure function of the brief and the request.

import {
  authHeaders,
  capabilitiesOf,
  joinBase,
  meetsRequirements,
  modelIsFree,
  ratingRowOf,
  type Probe,
  type ProviderEntry,
  type RatingsDoc,
  type Requirement,
  type Wire,
} from './brief.ts';
import {
  buildAnthropic,
  buildGemini,
  buildOpenAI,
  parseAnthropicJson,
  parseAnthropicStream,
  parseGeminiJson,
  parseGeminiStream,
  parseOpenAIJson,
  parseOpenAIStream,
  type Json,
  type NormAnswer,
  type NormRequest,
} from './wire.ts';

/** How long one provider call may take, in milliseconds. */
const UPSTREAM_TIMEOUT_MS = 20000;

export interface Candidate {
  provider: ProviderEntry;
  model: string;
  rating: number;
}

/**
 * The ranked candidates for one request: free models meeting its
 * requirements, best rating first. `jsonMode` is whether the client asked
 * for JSON mode (only ever true on the OpenAI wire).
 */
export function rankCandidates(
  probes: readonly Probe[],
  ratings: RatingsDoc | null,
  requires: Requirement,
  clientWire: 'openai' | 'anthropic',
  jsonMode: boolean,
): Candidate[] {
  const out: Candidate[] = [];
  for (const probe of probes) {
    if (!probe.ok) continue;
    // A JSON-mode request needs a wire that can carry it.
    if (jsonMode && probe.provider.wire === 'anthropic') continue;
    for (const model of probe.models) {
      const row = ratingRowOf(ratings, probe.provider.id, model.id);
      if (!modelIsFree(model, row)) continue;
      if (row === null) continue;
      if (!meetsRequirements(requires, capabilitiesOf(model, row))) continue;
      out.push({ provider: probe.provider, model: model.id, rating: row.rating_milli });
    }
  }
  out.sort((a, b) => {
    if (a.rating !== b.rating) return b.rating - a.rating;
    const aNative = a.provider.wire === clientWire ? 0 : 1;
    const bNative = b.provider.wire === clientWire ? 0 : 1;
    if (aNative !== bNative) return aNative - bNative;
    if (a.provider.id !== b.provider.id) return a.provider.id < b.provider.id ? -1 : 1;
    return a.model < b.model ? -1 : a.model > b.model ? 1 : 0;
  });
  return out;
}

/** The free models this entry serves, for GET /v1/models. */
export function servedModels(
  probes: readonly Probe[],
  ratings: RatingsDoc | null,
): { id: string; provider: string; display_name: string }[] {
  const out: { id: string; provider: string; display_name: string }[] = [];
  for (const probe of probes) {
    if (!probe.ok) continue;
    for (const model of [...probe.models].sort((a, b) => (a.id < b.id ? -1 : a.id > b.id ? 1 : 0))) {
      const row = ratingRowOf(ratings, probe.provider.id, model.id);
      if (!modelIsFree(model, row)) continue;
      out.push({ id: model.id, provider: probe.provider.id, display_name: model.display_name });
    }
  }
  return out;
}

export type UpstreamOutcome =
  | {
      kind: 'answer';
      candidate: Candidate;
      answer: NormAnswer;
    }
  | {
      kind: 'passthrough';
      candidate: Candidate;
      status: number;
      /** The provider's bytes, JSON or SSE, to send on as they are. */
      text: string;
      stream: boolean;
    }
  | { kind: 'failed' };

/** The request body and URL for one candidate's wire. */
function upstreamCall(
  envUrl: string,
  candidate: Candidate,
  norm: NormRequest,
): { url: string; body: Json } {
  const base = joinBase(envUrl, candidate.provider.base);
  const wire: Wire = candidate.provider.wire;
  if (wire === 'openai') {
    return { url: `${base}/v1/chat/completions`, body: buildOpenAI(norm, candidate.model) };
  }
  if (wire === 'anthropic') {
    return { url: `${base}/v1/messages`, body: buildAnthropic(norm, candidate.model) };
  }
  const action = norm.stream ? 'streamGenerateContent' : 'generateContent';
  const url = `${base}/v1beta/models/${encodeURIComponent(candidate.model)}:${action}${
    norm.stream ? '?alt=sse' : ''
  }`;
  return { url, body: buildGemini(norm) };
}

/** Parse one provider's answer into a normalized answer, by its wire and shape. */
function parseUpstream(
  wire: Wire,
  stream: boolean,
  text: string,
): NormAnswer | null {
  try {
    if (wire === 'openai') {
      if (stream) return parseOpenAIStream(text);
      const value: unknown = JSON.parse(text);
      return value !== null && typeof value === 'object' && !Array.isArray(value)
        ? parseOpenAIJson(value as Json)
        : null;
    }
    if (wire === 'anthropic') {
      if (stream) return parseAnthropicStream(text);
      const value: unknown = JSON.parse(text);
      return value !== null && typeof value === 'object' && !Array.isArray(value)
        ? parseAnthropicJson(value as Json)
        : null;
    }
    if (stream) return parseGeminiStream(text);
    const value: unknown = JSON.parse(text);
    return value !== null && typeof value === 'object' && !Array.isArray(value)
      ? parseGeminiJson(value as Json)
      : null;
  } catch {
    return null;
  }
}

/**
 * One attempt at one candidate: the parsed answer, or a pass-through when
 * the client and the provider share the OpenAI wire (the provider's bytes
 * are already a well-formed answer in the client's wire). A refusal or an
 * unreachable provider is 'next': the caller moves to the next candidate.
 * A 2xx that cannot be parsed is 'abort': the provider did answer, so
 * another success behind this request would break the trial's one-provider
 * rule, and the request is lost either way.
 */
async function attempt(
  envUrl: string,
  candidate: Candidate,
  norm: NormRequest,
  clientWire: 'openai' | 'anthropic',
): Promise<UpstreamOutcome | 'next' | 'abort'> {
  const { url, body } = upstreamCall(envUrl, candidate, norm);
  const headers: Record<string, string> = {
    'content-type': 'application/json',
    ...authHeaders(candidate.provider.auth, candidate.provider.key),
  };
  if (candidate.provider.wire === 'anthropic') headers['anthropic-version'] = '2023-06-01';
  if (norm.stream) headers.accept = 'text/event-stream';
  let res: Response;
  try {
    res = await fetch(url, {
      method: 'POST',
      headers,
      body: JSON.stringify(body),
      signal: AbortSignal.timeout(UPSTREAM_TIMEOUT_MS),
    });
  } catch {
    return 'next';
  }
  if (!res.ok) {
    // A refusal spends the call but is not an answer: drain and move on.
    try {
      await res.text();
    } catch {
      // The body is gone either way.
    }
    return 'next';
  }
  const text = await res.text();
  // OpenAI to OpenAI: the provider's bytes are already the client's
  // answer, JSON or SSE alike. Send them on untouched.
  if (clientWire === 'openai' && candidate.provider.wire === 'openai') {
    return { kind: 'passthrough', candidate, status: res.status, text, stream: norm.stream };
  }
  const answer = parseUpstream(candidate.provider.wire, norm.stream, text);
  if (answer === null) return 'abort';
  return { kind: 'answer', candidate, answer };
}

/**
 * Serve one request: try the candidates in rank order until one answers.
 * Returns the first success, or `failed` when no candidate answers.
 */
export async function serveRequest(
  envUrl: string,
  candidates: readonly Candidate[],
  norm: NormRequest,
  clientWire: 'openai' | 'anthropic',
): Promise<UpstreamOutcome> {
  for (const candidate of candidates) {
    const outcome = await attempt(envUrl, candidate, norm, clientWire);
    if (outcome === 'next') continue;
    if (outcome === 'abort') return { kind: 'failed' };
    return outcome;
  }
  return { kind: 'failed' };
}
