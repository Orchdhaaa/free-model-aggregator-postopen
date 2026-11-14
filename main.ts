// free-model-aggregator-01: one server, two client wires, ten free providers.
//
// The server reads the trial brief, probes each provider's catalog route
// in the provider's own wire, builds one catalog, registers its port and
// catalog with the env, then serves the client's requests: GET /v1/models
// in the shape of the auth header it got (x-api-key means Anthropic,
// Authorization means OpenAI), POST /v1/chat/completions and POST
// /v1/messages, each as JSON or server-sent events. It polls GET /work
// every 100 ms and exits 0 the moment the env says the workload is done.
// A process still running when the trial's wall ends scores zero, so it
// never waits for anything else.
//
// Routing: for each request the free models meeting its requirements are
// ranked by the sealed quality table's rating, best first, and tried in
// order until one answers. The manifest's row for the arriving request
// (in arrival order) names its requirements; a request past the manifest
// reads its requirements off its own body. Every routing choice is a
// function of the brief, the requests and the providers' answers: no wall
// clock and no unseeded randomness, so a replay reproduces the trial.
//
// `node main.ts --list` prints the same catalog JSON it registers, byte
// for byte, and exits 0 before any server starts. Run outside a trial (no
// MAKEFUN_PROMPT_FILE, no MAKEFUN_ENV_URL) it prints {"providers":[]}
// and exits 0.

import { createServer, type IncomingMessage, type Server, type ServerResponse } from 'node:http';
import {
  buildCatalog,
  isRecord,
  probeProvider,
  readBrief,
  type CatalogReport,
  type Json,
  type Probe,
  type Requirement,
} from './lib/brief.ts';
import {
  anthropicError,
  anthropicModelList,
  openAIError,
  openAIModelList,
  parseAnthropic,
  parseOpenAI,
  renderAnthropicJson,
  renderAnthropicStream,
  renderOpenAIJson,
  renderOpenAIStream,
  type NormRequest,
} from './lib/wire.ts';
import { rankCandidates, serveRequest, servedModels } from './lib/upstream.ts';

/** The wire protocol's version, in the register body. */
const PROTOCOL_VERSION = 1;
/** How often GET /work is polled, in milliseconds. */
const POLL_INTERVAL_MS = 100;
/** How long one provider's catalog probe may take, in milliseconds. */
const PROBE_TIMEOUT_MS = 5000;
/** The most bytes of one client request body this server reads. */
const MAX_BODY_BYTES = 32 * 1024 * 1024;

const ENV_URL = (process.env.MAKEFUN_ENV_URL ?? '').replace(/\/+$/, '');
const PROMPT_FILE = process.env.MAKEFUN_PROMPT_FILE ?? '';

const brief = readBrief(PROMPT_FILE);
let probes: Probe[] = [];
let catalog: CatalogReport = { providers: [] };
/** How many client API requests have arrived: the manifest is in arrival order. */
let manifestIndex = 0;

function log(message: string): void {
  process.stderr.write(`aggregator: ${message}\n`);
}

/** Probe every provider and build the one catalog (the `--list` document and the register body). */
async function load(): Promise<void> {
  if (brief.error !== null) log(brief.error);
  probes = await Promise.all(
    brief.providers.map((provider) => probeProvider(ENV_URL, provider, PROBE_TIMEOUT_MS)),
  );
  for (const probe of probes) if (!probe.ok && probe.error !== null) log(probe.error);
  catalog = buildCatalog(probes, brief.ratings);
}

/** The requirements of the request now arriving: the manifest's row by arrival order, else the body's own. */
function requirementsFor(body: Json, clientWire: 'openai' | 'anthropic'): Requirement {
  const manifest = brief.workload;
  const entry = manifest[manifestIndex];
  manifestIndex += 1;
  if (entry !== undefined) return entry.requires;
  // The manifest is the declared source of each request's requirements;
  // this fallback only guesses for a brief that holds none.
  return requirementsFromBody(body, clientWire);
}

/** Requirements read off a request body, when the manifest has no row for it. */
function requirementsFromBody(body: Json, clientWire: 'openai' | 'anthropic'): Requirement {
  const norm: NormRequest =
    clientWire === 'openai' ? parseOpenAI(body) : parseAnthropic(body);
  const requires: Requirement = {};
  if (norm.tools.length > 0) requires.tools = true;
  if (norm.messages.some((m) => m.images.length > 0)) requires.vision = true;
  if (norm.jsonMode) requires.json = true;
  return requires;
}

/** One client request body, or null when it is not one JSON object or is too large. */
async function readJson(req: IncomingMessage): Promise<Json | null> {
  const chunks: Buffer[] = [];
  let size = 0;
  for await (const chunk of req) {
    const buffer = Buffer.isBuffer(chunk) ? chunk : Buffer.from(String(chunk));
    size += buffer.length;
    if (size > MAX_BODY_BYTES) return null;
    chunks.push(buffer);
  }
  if (chunks.length === 0) return null;
  try {
    const value: unknown = JSON.parse(Buffer.concat(chunks).toString('utf8'));
    return isRecord(value) ? value : null;
  } catch {
    return null;
  }
}

function sendJson(res: ServerResponse, status: number, body: unknown): void {
  const text = JSON.stringify(body);
  res.writeHead(status, {
    'content-type': 'application/json',
    'content-length': Buffer.byteLength(text),
  });
  res.end(text);
}

function sendText(
  res: ServerResponse,
  status: number,
  text: string,
  contentType: string,
): void {
  res.writeHead(status, {
    'content-type': contentType,
    'content-length': Buffer.byteLength(text),
  });
  res.end(text);
}

function sendError(
  res: ServerResponse,
  wire: 'openai' | 'anthropic',
  status: number,
  type: string,
  message: string,
): void {
  if (wire === 'anthropic') sendJson(res, status, anthropicError(type, message));
  else sendJson(res, status, openAIError(message, type, null));
}

/** GET /v1/models, in the shape of the auth header the client sent. */
function models(req: IncomingMessage, res: ServerResponse): void {
  const list = servedModels(probes, brief.ratings);
  if (typeof req.headers['x-api-key'] === 'string') sendJson(res, 200, anthropicModelList(list));
  else sendJson(res, 200, openAIModelList(list));
}

/**
 * Serve one client request on one wire: parse it, rank the candidates
 * that meet its requirements, try them in order, and render the first
 * answer in the client's wire (or pass an OpenAI provider's bytes on
 * untouched when the client speaks OpenAI too).
 */
async function serve(
  req: IncomingMessage,
  res: ServerResponse,
  clientWire: 'openai' | 'anthropic',
): Promise<void> {
  const body = await readJson(req);
  if (body === null) {
    return sendError(
      res,
      clientWire,
      400,
      clientWire === 'anthropic' ? 'invalid_request_error' : 'invalid_request_error',
      'the request body is not one JSON object',
    );
  }
  const norm: NormRequest = clientWire === 'openai' ? parseOpenAI(body) : parseAnthropic(body);
  const requires = requirementsFor(body, clientWire);
  const candidates = rankCandidates(probes, brief.ratings, requires, clientWire, norm.jsonMode);
  if (candidates.length === 0) {
    return sendError(
      res,
      clientWire,
      503,
      clientWire === 'anthropic' ? 'overloaded_error' : 'no_free_model',
      'no free model on the directory meets this request',
    );
  }
  const outcome = await serveRequest(ENV_URL, candidates, norm, clientWire);
  if (outcome.kind === 'failed') {
    return sendError(
      res,
      clientWire,
      502,
      clientWire === 'anthropic' ? 'api_error' : 'upstream_unreachable',
      'every free model that meets this request refused it',
    );
  }
  const model = outcome.candidate.model;
  if (outcome.kind === 'passthrough') {
    const contentType = outcome.stream
      ? 'text/event-stream; charset=utf-8'
      : 'application/json';
    return sendText(res, outcome.status, outcome.text, contentType);
  }
  if (clientWire === 'openai') {
    if (norm.stream) {
      return sendText(res, 200, renderOpenAIStream(outcome.answer, model), 'text/event-stream; charset=utf-8');
    }
    return sendJson(res, 200, renderOpenAIJson(outcome.answer, model));
  }
  if (norm.stream) {
    return sendText(res, 200, renderAnthropicStream(outcome.answer, model), 'text/event-stream; charset=utf-8');
  }
  return sendJson(res, 200, renderAnthropicJson(outcome.answer, model));
}

async function handle(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const url = new URL(req.url ?? '/', 'http://entry');
  if (req.method === 'GET' && url.pathname === '/v1/models') {
    models(req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/chat/completions') {
    await serve(req, res, 'openai');
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/messages') {
    await serve(req, res, 'anthropic');
    return;
  }
  sendError(
    res,
    url.pathname === '/v1/messages' ? 'anthropic' : 'openai',
    404,
    'not_found',
    `no route ${req.method ?? ''} ${url.pathname}`,
  );
}

function listen(server: Server): Promise<void> {
  return new Promise((resolve, reject) => {
    server.once('error', reject);
    server.listen(0, '0.0.0.0', () => resolve());
  });
}

/** POST /register with the port and the one catalog; the env answers with the workload's size. */
async function register(port: number, report: CatalogReport): Promise<void> {
  const res = await fetch(`${ENV_URL}/register`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ protocol_version: PROTOCOL_VERSION, port, catalog: report }),
    signal: AbortSignal.timeout(5000),
  });
  if (!res.ok) {
    throw new Error(`POST ${ENV_URL}/register answered HTTP ${res.status}: ${(await res.text()).slice(0, 500)}`);
  }
  const reply: unknown = await res.json();
  if (isRecord(reply) && typeof reply.requests === 'number') {
    log(`registered: the workload holds ${reply.requests} requests`);
  }
}

/** Poll GET /work every 100 ms; exit 0 the moment it says done. */
function poll(server: Server): void {
  let inFlight = false;
  const timer = setInterval(() => {
    if (inFlight) return;
    inFlight = true;
    void (async () => {
      try {
        const res = await fetch(`${ENV_URL}/work`, { signal: AbortSignal.timeout(2000) });
        if (res.status !== 200) return;
        const work: unknown = await res.json();
        if (isRecord(work) && work.done === true) {
          clearInterval(timer);
          server.close();
          process.exit(0);
        }
      } catch {
        // The env is not reachable yet: try again on the next tick.
      } finally {
        inFlight = false;
      }
    })();
  }, POLL_INTERVAL_MS);
}

async function start(): Promise<void> {
  await load();
  if (process.argv.includes('--list')) {
    // The same document the register body carries, byte for byte: print
    // the one built object and stop before any server starts. A process
    // whose stdout is not flushed loses the tail, so wait for it.
    const text = `${JSON.stringify(catalog)}\n`;
    await new Promise<void>((resolve) => process.stdout.write(text, () => resolve()));
    process.exit(0);
  }
  if (ENV_URL === '') throw new Error('MAKEFUN_ENV_URL is not set');
  const server = createServer((req, res) => {
    void handle(req, res).catch((err: unknown) => {
      log((err as Error).stack ?? String(err));
      if (!res.headersSent) {
        sendError(
          res,
          req.url?.startsWith('/v1/messages') === true ? 'anthropic' : 'openai',
          500,
          'internal_error',
          'the server failed to answer',
        );
      }
    });
  });
  await listen(server);
  const address = server.address();
  if (address === null || typeof address === 'string') throw new Error('the server has no port');
  const port = address.port;
  // 8080 is the env's on one layout; the OS's ephemeral range never holds
  // it, and the guard says so.
  if (port === 8080) throw new Error('the OS picked 8080, which the env owns on one layout: restart the trial');
  const freeCount = servedModels(probes, brief.ratings).length;
  log(`listening on 0.0.0.0:${port} with ${catalog.providers.length} answered providers and ${freeCount} free models`);
  await register(port, catalog);
  process.on('SIGTERM', () => process.exit(0));
  process.on('SIGINT', () => process.exit(0));
  poll(server);
}

try {
  await start();
} catch (err) {
  log((err as Error).stack ?? String(err));
  process.exit(1);
}
