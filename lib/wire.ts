// The client wires and the three upstream wires.
//
// The trial's client calls this entry as an OpenAI service (POST
// /v1/chat/completions) or an Anthropic one (POST /v1/messages), with a
// JSON body or with server-sent events. Every request is first parsed into
// one normalized shape (system text, messages, tools, caps, stops, JSON
// mode), then built into whichever upstream wire serves it, then the
// provider's answer is parsed back into one normalized answer (text, tool
// calls, stop family, usage) and rendered in the wire the client used.
//
// The normalized shapes are pure functions of their arguments: no clock
// and no randomness, so the same body and the same provider answer give
// the same bytes on a replay. The model listings use fixed timestamps for
// the same reason.
//
// Stop reasons travel as families (0 stop, 1 length, 2 tool use, 3
// refusal) because the three wires spell the same outcome differently:
// OpenAI stop/length/tool_calls/content_filter, Anthropic
// end_turn/max_tokens/tool_use/refusal, Gemini STOP/MAX_TOKENS/SAFETY.

import { isRecord, type Json } from './brief.ts';

/** A fixed epoch second for every listed model's `created`. */
export const FIXED_CREATED = 1767225600;
/** A fixed instant for every listed model's `created_at`. */
export const FIXED_CREATED_AT = '2026-01-01T00:00:00Z';

/** One served model, as GET /v1/models lists it. */
export interface ServedModel {
  id: string;
  provider: string;
  display_name: string;
}

/** GET /v1/models on an OpenAI-shaped request. */
export function openAIModelList(models: readonly ServedModel[]): Json {
  return {
    object: 'list',
    data: models.map((m) => ({
      id: m.id,
      object: 'model',
      created: FIXED_CREATED,
      owned_by: m.provider,
    })),
  };
}

/** GET /v1/models on an Anthropic-shaped request (an x-api-key header). */
export function anthropicModelList(models: readonly ServedModel[]): Json {
  return {
    data: models.map((m) => ({
      type: 'model',
      id: m.id,
      display_name: m.display_name,
      created_at: FIXED_CREATED_AT,
    })),
    has_more: false,
  };
}

/** An error in Anthropic's shape (the Messages API's error envelope). */
export function anthropicError(type: string, message: string): Json {
  return { type: 'error', error: { type, message } };
}

/** An error in OpenAI's shape (Chat Completions). */
export function openAIError(message: string, type: string, code: string | null): Json {
  return { error: { message, type, code } };
}

// ---------------------------------------------------------------------------
// Normalized request
// ---------------------------------------------------------------------------

export interface NormImage {
  /** "data" for an inline base64 image, "url" for a remote one. */
  kind: 'data' | 'url';
  mediaType: string;
  data: string;
}

export interface NormToolCall {
  id: string;
  name: string;
  /** Arguments as a JSON string (what the wire carries). */
  args: string;
}

export interface NormMessage {
  role: 'user' | 'assistant' | 'tool';
  text: string;
  images: NormImage[];
  toolCalls: NormToolCall[];
  /** For a tool-role message: the call id its result answers. */
  toolCallId: string;
}

export interface NormTool {
  name: string;
  description: string;
  parameters: Json;
}

export interface NormRequest {
  system: string;
  messages: NormMessage[];
  tools: NormTool[];
  /** 'auto' | 'none' | { name } | undefined */
  toolChoice: 'auto' | 'none' | { name: string } | undefined;
  maxTokens: number | null;
  temperature: number | undefined;
  topP: number | undefined;
  stops: string[];
  jsonMode: boolean;
  stream: boolean;
}

function numberOrNull(v: unknown): number | null {
  return typeof v === 'number' && Number.isFinite(v) ? v : null;
}

function numberOrZero(v: unknown): number {
  return typeof v === 'number' && Number.isFinite(v) ? v : 0;
}

/** The text of a string or of a block array's text blocks. */
export function textOf(content: unknown): string {
  if (typeof content === 'string') return content;
  if (!Array.isArray(content)) return '';
  return content
    .flatMap((b) => (isRecord(b) && b.type === 'text' && typeof b.text === 'string' ? [b.text] : []))
    .join('\n');
}

function stringList(v: unknown): string[] {
  if (typeof v === 'string') return [v];
  if (!Array.isArray(v)) return [];
  return v.filter((x): x is string => typeof x === 'string');
}

/** One OpenAI content part as a normalized image, or null. */
function imageOfPart(part: Json): NormImage | null {
  if (part.type !== 'image_url' || !isRecord(part.image_url)) return null;
  const url = typeof part.image_url.url === 'string' ? part.image_url.url : '';
  if (url === '') return null;
  const data = /^data:([^;,]*);base64,(.*)$/s.exec(url);
  if (data !== null) {
    return { kind: 'data', mediaType: data[1] === '' ? 'image/png' : data[1], data: data[2] };
  }
  return { kind: 'url', mediaType: '', data: url };
}

/** An OpenAI Chat Completions request as a normalized request. */
export function parseOpenAI(body: Json): NormRequest {
  let system = '';
  const messages: NormMessage[] = [];
  for (const raw of Array.isArray(body.messages) ? body.messages : []) {
    if (!isRecord(raw)) continue;
    const role = raw.role;
    if (role === 'system') {
      const text = textOf(raw.content);
      if (text !== '') system = system === '' ? text : `${system}\n${text}`;
      continue;
    }
    if (role === 'assistant') {
      const toolCalls: NormToolCall[] = [];
      for (const call of Array.isArray(raw.tool_calls) ? raw.tool_calls : []) {
        if (!isRecord(call) || !isRecord(call.function)) continue;
        toolCalls.push({
          id: typeof call.id === 'string' ? call.id : '',
          name: typeof call.function.name === 'string' ? call.function.name : '',
          args:
            typeof call.function.arguments === 'string'
              ? call.function.arguments
              : JSON.stringify(call.function.arguments ?? {}),
        });
      }
      messages.push({
        role: 'assistant',
        text: textOf(raw.content),
        images: [],
        toolCalls,
        toolCallId: '',
      });
      continue;
    }
    if (role === 'tool') {
      messages.push({
        role: 'tool',
        text: typeof raw.content === 'string' ? raw.content : textOf(raw.content),
        images: [],
        toolCalls: [],
        toolCallId: typeof raw.tool_call_id === 'string' ? raw.tool_call_id : '',
      });
      continue;
    }
    if (role !== 'user') continue;
    const images: NormImage[] = [];
    if (Array.isArray(raw.content)) {
      for (const part of raw.content) {
        if (!isRecord(part)) continue;
        const image = imageOfPart(part);
        if (image !== null) images.push(image);
      }
    }
    messages.push({ role: 'user', text: textOf(raw.content), images, toolCalls: [], toolCallId: '' });
  }
  const tools: NormTool[] = [];
  for (const raw of Array.isArray(body.tools) ? body.tools : []) {
    if (!isRecord(raw) || !isRecord(raw.function) || typeof raw.function.name !== 'string') continue;
    tools.push({
      name: raw.function.name,
      description: typeof raw.function.description === 'string' ? raw.function.description : '',
      parameters: isRecord(raw.function.parameters)
        ? raw.function.parameters
        : { type: 'object', properties: {} },
    });
  }
  let toolChoice: NormRequest['toolChoice'];
  const choice = body.tool_choice;
  if (choice === 'auto' || choice === 'none' || choice === 'required') {
    toolChoice = choice === 'required' ? 'auto' : choice;
  } else if (isRecord(choice) && choice.type === 'function' && isRecord(choice.function)) {
    toolChoice = { name: typeof choice.function.name === 'string' ? choice.function.name : '' };
  }
  const format = isRecord(body.response_format) ? body.response_format.type : undefined;
  return {
    system,
    messages,
    tools,
    toolChoice,
    maxTokens: numberOrNull(body.max_tokens) ?? numberOrNull(body.max_completion_tokens),
    temperature: typeof body.temperature === 'number' ? body.temperature : undefined,
    topP: typeof body.top_p === 'number' ? body.top_p : undefined,
    stops: stringList(body.stop),
    jsonMode: format === 'json_object' || format === 'json_schema',
    stream: body.stream === true,
  };
}

/** A tool_result block's content as text. */
function toolResultText(content: unknown): string {
  if (typeof content === 'string') return content;
  if (Array.isArray(content)) {
    const text = textOf(content);
    return text !== '' ? text : JSON.stringify(content);
  }
  return content === undefined ? '' : JSON.stringify(content);
}

/** An Anthropic Messages request as a normalized request. */
export function parseAnthropic(body: Json): NormRequest {
  const system = textOf(body.system);
  const messages: NormMessage[] = [];
  for (const raw of Array.isArray(body.messages) ? body.messages : []) {
    if (!isRecord(raw)) continue;
    const assistant = raw.role === 'assistant';
    const content = raw.content;
    if (typeof content === 'string') {
      messages.push({
        role: assistant ? 'assistant' : 'user',
        text: content,
        images: [],
        toolCalls: [],
        toolCallId: '',
      });
      continue;
    }
    if (!Array.isArray(content)) continue;
    if (assistant) {
      const toolCalls: NormToolCall[] = [];
      for (const block of content) {
        if (!isRecord(block) || block.type !== 'tool_use') continue;
        toolCalls.push({
          id: typeof block.id === 'string' ? block.id : '',
          name: typeof block.name === 'string' ? block.name : '',
          args: JSON.stringify(isRecord(block.input) ? block.input : {}),
        });
      }
      messages.push({
        role: 'assistant',
        text: textOf(content),
        images: [],
        toolCalls,
        toolCallId: '',
      });
      continue;
    }
    const images: NormImage[] = [];
    let toolCallId = '';
    let toolText = '';
    let isTool = false;
    for (const block of content) {
      if (!isRecord(block)) continue;
      if (block.type === 'tool_result') {
        isTool = true;
        toolCallId = typeof block.tool_use_id === 'string' ? block.tool_use_id : '';
        toolText = toolResultText(block.content);
        continue;
      }
      if (block.type === 'image' && isRecord(block.source)) {
        const source = block.source;
        if (source.type === 'base64' && typeof source.data === 'string') {
          images.push({
            kind: 'data',
            mediaType: typeof source.media_type === 'string' ? source.media_type : 'image/png',
            data: source.data,
          });
        } else if (source.type === 'url' && typeof source.url === 'string') {
          images.push({ kind: 'url', mediaType: '', data: source.url });
        }
      }
    }
    if (isTool) {
      messages.push({ role: 'tool', text: toolText, images: [], toolCalls: [], toolCallId });
    } else {
      messages.push({ role: 'user', text: textOf(content), images, toolCalls: [], toolCallId: '' });
    }
  }
  const tools: NormTool[] = [];
  for (const raw of Array.isArray(body.tools) ? body.tools : []) {
    if (!isRecord(raw) || typeof raw.name !== 'string') continue;
    tools.push({
      name: raw.name,
      description: typeof raw.description === 'string' ? raw.description : '',
      parameters: isRecord(raw.input_schema) ? raw.input_schema : { type: 'object', properties: {} },
    });
  }
  let toolChoice: NormRequest['toolChoice'];
  const choice = body.tool_choice;
  if (isRecord(choice)) {
    if (choice.type === 'auto') toolChoice = 'auto';
    else if (choice.type === 'any') toolChoice = 'auto';
    else if (choice.type === 'none') toolChoice = 'none';
    else if (choice.type === 'tool' && typeof choice.name === 'string') toolChoice = { name: choice.name };
  }
  return {
    system,
    messages,
    tools,
    toolChoice,
    maxTokens: numberOrNull(body.max_tokens),
    temperature: typeof body.temperature === 'number' ? body.temperature : undefined,
    topP: typeof body.top_p === 'number' ? body.top_p : undefined,
    stops: stringList(body.stop_sequences),
    // The Anthropic wire carries no JSON mode: a client on this wire never
    // asks for it, so no upstream wire needs to carry it for one.
    jsonMode: false,
    stream: body.stream === true,
  };
}

// ---------------------------------------------------------------------------
// Building upstream requests
// ---------------------------------------------------------------------------

/** A normalized request as an OpenAI Chat Completions body naming `model`. */
export function buildOpenAI(norm: NormRequest, model: string): Json {
  const messages: Json[] = [];
  if (norm.system !== '') messages.push({ role: 'system', content: norm.system });
  for (const m of norm.messages) {
    if (m.role === 'assistant') {
      const out: Json = { role: 'assistant', content: m.text !== '' ? m.text : null };
      if (m.toolCalls.length > 0) {
        out.tool_calls = m.toolCalls.map((c) => ({
          id: c.id,
          type: 'function',
          function: { name: c.name, arguments: c.args },
        }));
      }
      messages.push(out);
      continue;
    }
    if (m.role === 'tool') {
      messages.push({ role: 'tool', tool_call_id: m.toolCallId, content: m.text });
      continue;
    }
    const parts: Json[] = [];
    if (m.text !== '') parts.push({ type: 'text', text: m.text });
    for (const image of m.images) {
      parts.push({
        type: 'image_url',
        image_url: {
          url:
            image.kind === 'data'
              ? `data:${image.mediaType};base64,${image.data}`
              : image.data,
        },
      });
    }
    messages.push({ role: 'user', content: parts.length === 1 && m.text !== '' && m.images.length === 0 ? m.text : parts });
  }
  const out: Json = { model, messages };
  if (norm.maxTokens !== null) out.max_tokens = norm.maxTokens;
  if (norm.temperature !== undefined) out.temperature = norm.temperature;
  if (norm.topP !== undefined) out.top_p = norm.topP;
  if (norm.stops.length > 0) out.stop = norm.stops;
  if (norm.tools.length > 0) {
    out.tools = norm.tools.map((t) => ({
      type: 'function',
      function: { name: t.name, description: t.description, parameters: t.parameters },
    }));
  }
  if (norm.toolChoice === 'auto') out.tool_choice = 'auto';
  else if (norm.toolChoice === 'none') out.tool_choice = 'none';
  else if (norm.toolChoice !== undefined) {
    out.tool_choice = { type: 'function', function: { name: norm.toolChoice.name } };
  }
  if (norm.jsonMode) out.response_format = { type: 'json_object' };
  if (norm.stream) {
    out.stream = true;
    // Ask the provider to close the stream with a usage chunk: the
    // client's stream parser reads the usage from it.
    out.stream_options = { include_usage: true };
  }
  return out;
}

/** A normalized request as an Anthropic Messages body naming `model`. */
export function buildAnthropic(norm: NormRequest, model: string): Json {
  const messages: Json[] = [];
  for (const m of norm.messages) {
    if (m.role === 'assistant') {
      const content: Json[] = [];
      if (m.text !== '') content.push({ type: 'text', text: m.text });
      for (const call of m.toolCalls) {
        content.push({
          type: 'tool_use',
          id: call.id,
          name: call.name,
          input: parseArgsObject(call.args),
        });
      }
      messages.push({ role: 'assistant', content });
      continue;
    }
    if (m.role === 'tool') {
      messages.push({
        role: 'user',
        content: [{ type: 'tool_result', tool_use_id: m.toolCallId, content: m.text }],
      });
      continue;
    }
    const content: Json[] = [];
    if (m.text !== '') content.push({ type: 'text', text: m.text });
    for (const image of m.images) {
      content.push({
        type: 'image',
        source:
          image.kind === 'data'
            ? { type: 'base64', media_type: image.mediaType, data: image.data }
            : { type: 'url', url: image.data },
      });
    }
    messages.push({ role: 'user', content });
  }
  const out: Json = { model, messages };
  if (norm.system !== '') out.system = norm.system;
  if (norm.maxTokens !== null) out.max_tokens = norm.maxTokens;
  if (norm.temperature !== undefined) out.temperature = norm.temperature;
  if (norm.topP !== undefined) out.top_p = norm.topP;
  if (norm.stops.length > 0) out.stop_sequences = norm.stops;
  if (norm.tools.length > 0) {
    out.tools = norm.tools.map((t) => ({
      name: t.name,
      description: t.description,
      input_schema: t.parameters,
    }));
  }
  if (norm.toolChoice === 'auto') out.tool_choice = { type: 'auto' };
  else if (norm.toolChoice === 'none') out.tool_choice = { type: 'none' };
  else if (norm.toolChoice !== undefined) {
    out.tool_choice = { type: 'tool', name: norm.toolChoice.name };
  }
  if (norm.stream) out.stream = true;
  return out;
}

/** A normalized request as a Gemini generateContent body (model goes in the URL). */
export function buildGemini(norm: NormRequest): Json {
  const contents: Json[] = [];
  for (const m of norm.messages) {
    if (m.role === 'assistant') {
      const parts: Json[] = [];
      if (m.text !== '') parts.push({ text: m.text });
      for (const call of m.toolCalls) {
        parts.push({
          functionCall: { name: call.name, args: parseArgsObject(call.args) },
        });
      }
      contents.push({ role: 'model', parts });
      continue;
    }
    if (m.role === 'tool') {
      contents.push({
        role: 'user',
        parts: [
          {
            functionResponse: {
              name: '',
              response: { content: m.text },
            },
          },
        ],
      });
      continue;
    }
    const parts: Json[] = [];
    if (m.text !== '') parts.push({ text: m.text });
    for (const image of m.images) {
      if (image.kind === 'data') {
        parts.push({ inlineData: { mimeType: image.mediaType, data: image.data } });
      } else {
        parts.push({ fileData: { mimeType: 'image/png', fileUri: image.data } });
      }
    }
    contents.push({ role: 'user', parts });
  }
  const out: Json = { contents };
  if (norm.system !== '') {
    out.systemInstruction = { parts: [{ text: norm.system }] };
  }
  if (norm.tools.length > 0) {
    out.tools = [
      {
        functionDeclarations: norm.tools.map((t) => ({
          name: t.name,
          description: t.description,
          parameters: t.parameters,
        })),
      },
    ];
  }
  const config: Json = {};
  if (norm.maxTokens !== null) config.maxOutputTokens = norm.maxTokens;
  if (norm.stops.length > 0) config.stopSequences = norm.stops;
  if (norm.jsonMode) config.responseMimeType = 'application/json';
  if (norm.temperature !== undefined) config.temperature = norm.temperature;
  if (norm.topP !== undefined) config.topP = norm.topP;
  if (Object.keys(config).length > 0) out.generationConfig = config;
  return out;
}

/** A tool call's arguments as an object: the JSON string, or {} when it is not one. */
export function parseArgsObject(value: unknown): Json {
  if (isRecord(value)) return value;
  if (typeof value === 'string' && value.trim() !== '') {
    try {
      const parsed: unknown = JSON.parse(value);
      if (isRecord(parsed)) return parsed;
    } catch {
      // Not JSON: an empty input.
    }
  }
  return {};
}

// ---------------------------------------------------------------------------
// Normalized answer
// ---------------------------------------------------------------------------

/** Stop families: 0 stop, 1 length, 2 tool use, 3 refusal. */
export type StopFamily = 0 | 1 | 2 | 3;

export interface NormAnswer {
  text: string;
  toolCalls: NormToolCall[];
  stopFamily: StopFamily;
  usage: { input: number; output: number };
}

function openAIFamily(finish: unknown): StopFamily {
  if (finish === 'length') return 1;
  if (finish === 'tool_calls' || finish === 'function_call') return 2;
  if (finish === 'content_filter') return 3;
  return 0;
}

function anthropicFamily(stop: unknown): StopFamily {
  if (stop === 'max_tokens') return 1;
  if (stop === 'tool_use') return 2;
  if (stop === 'refusal') return 3;
  return 0;
}

function geminiFamily(finish: unknown): StopFamily {
  if (finish === 'MAX_TOKENS') return 1;
  if (finish === 'SAFETY') return 3;
  return 0;
}

/** One OpenAI Chat Completions JSON answer as a normalized answer. */
export function parseOpenAIJson(body: Json): NormAnswer {
  const choices = Array.isArray(body.choices) ? body.choices : [];
  const choice = isRecord(choices[0]) ? choices[0] : {};
  const message = isRecord(choice.message) ? choice.message : {};
  const toolCalls: NormToolCall[] = [];
  for (const call of Array.isArray(message.tool_calls) ? message.tool_calls : []) {
    if (!isRecord(call) || !isRecord(call.function)) continue;
    toolCalls.push({
      id: typeof call.id === 'string' ? call.id : '',
      name: typeof call.function.name === 'string' ? call.function.name : '',
      args:
        typeof call.function.arguments === 'string'
          ? call.function.arguments
          : JSON.stringify(call.function.arguments ?? {}),
    });
  }
  const usage = isRecord(body.usage) ? body.usage : {};
  return {
    text: typeof message.content === 'string' ? message.content : textOf(message.content),
    toolCalls,
    stopFamily: openAIFamily(choice.finish_reason),
    usage: {
      input: numberOrZero(usage.prompt_tokens),
      output: numberOrZero(usage.completion_tokens),
    },
  };
}

/** One SSE frame's lines: the event name (or null) and its data lines joined. */
export interface SseFrame {
  event: string | null;
  data: string;
}

export function parseSseFrames(text: string): SseFrame[] {
  const frames: SseFrame[] = [];
  const normalized = text.replace(/\r\n/g, '\n');
  for (const raw of normalized.split('\n\n')) {
    if (raw.trim() === '') continue;
    let event: string | null = null;
    const data: string[] = [];
    for (const line of raw.split('\n')) {
      if (line.startsWith('event:')) event = line.slice('event:'.length).trim();
      else if (line.startsWith('data:')) data.push(line.slice('data:'.length).replace(/^ /, ''));
    }
    if (data.length > 0) frames.push({ event, data: data.join('\n') });
  }
  return frames;
}

function parseJsonLine(payload: string): Json | null {
  try {
    const value: unknown = JSON.parse(payload);
    return isRecord(value) ? value : null;
  } catch {
    return null;
  }
}

/** One OpenAI SSE stream as a normalized answer. */
export function parseOpenAIStream(text: string): NormAnswer {
  let outText = '';
  const calls = new Map<number, NormToolCall>();
  let stopFamily: StopFamily = 0;
  let input = 0;
  let output = 0;
  for (const frame of parseSseFrames(text)) {
    const payload = frame.data.trim();
    if (payload === '[DONE]') continue;
    const chunk = parseJsonLine(payload);
    if (chunk === null) continue;
    if (isRecord(chunk.usage)) {
      input = numberOrZero(chunk.usage.prompt_tokens);
      output = numberOrZero(chunk.usage.completion_tokens);
    }
    const choices = Array.isArray(chunk.choices) ? chunk.choices : [];
    if (choices.length === 0) continue;
    const choice = isRecord(choices[0]) ? choices[0] : null;
    if (choice === null) continue;
    const delta = isRecord(choice.delta) ? choice.delta : {};
    if (typeof delta.content === 'string') outText += delta.content;
    for (const call of Array.isArray(delta.tool_calls) ? delta.tool_calls : []) {
      if (!isRecord(call)) continue;
      const index = typeof call.index === 'number' ? call.index : 0;
      const at = calls.get(index) ?? { id: '', name: '', args: '' };
      if (typeof call.id === 'string' && call.id !== '') at.id = call.id;
      const fn = isRecord(call.function) ? call.function : {};
      if (typeof fn.name === 'string' && fn.name !== '') at.name = fn.name;
      if (typeof fn.arguments === 'string') at.args += fn.arguments;
      calls.set(index, at);
    }
    if (typeof choice.finish_reason === 'string') stopFamily = openAIFamily(choice.finish_reason);
  }
  return {
    text: outText,
    toolCalls: [...calls.entries()].sort((a, b) => a[0] - b[0]).map(([, c]) => c),
    stopFamily,
    usage: { input, output },
  };
}

/** One Anthropic Messages JSON answer as a normalized answer. */
export function parseAnthropicJson(body: Json): NormAnswer {
  const content = Array.isArray(body.content) ? body.content : [];
  let outText = '';
  const toolCalls: NormToolCall[] = [];
  for (const block of content) {
    if (!isRecord(block)) continue;
    if (block.type === 'text' && typeof block.text === 'string') outText += block.text;
    else if (block.type === 'tool_use') {
      toolCalls.push({
        id: typeof block.id === 'string' ? block.id : '',
        name: typeof block.name === 'string' ? block.name : '',
        args: JSON.stringify(isRecord(block.input) ? block.input : {}),
      });
    }
  }
  const usage = isRecord(body.usage) ? body.usage : {};
  return {
    text: outText,
    toolCalls,
    stopFamily: anthropicFamily(body.stop_reason),
    usage: {
      input: numberOrZero(usage.input_tokens),
      output: numberOrZero(usage.output_tokens),
    },
  };
}

/** One Anthropic SSE stream as a normalized answer. */
export function parseAnthropicStream(text: string): NormAnswer {
  let outText = '';
  const blocks = new Map<number, { kind: 'text' | 'tool'; id: string; name: string; args: string }>();
  let stopFamily: StopFamily = 0;
  let input = 0;
  let output = 0;
  for (const frame of parseSseFrames(text)) {
    const data = parseJsonLine(frame.data);
    if (data === null) continue;
    switch (frame.event) {
      case 'message_start': {
        if (isRecord(data.message) && isRecord(data.message.usage)) {
          input = numberOrZero(data.message.usage.input_tokens);
        }
        break;
      }
      case 'content_block_start': {
        const index = numberOrNull(data.index);
        if (index === null || !isRecord(data.content_block)) break;
        const block = data.content_block;
        if (block.type === 'text') {
          blocks.set(index, { kind: 'text', id: '', name: '', args: '' });
        } else if (block.type === 'tool_use') {
          blocks.set(index, {
            kind: 'tool',
            id: typeof block.id === 'string' ? block.id : '',
            name: typeof block.name === 'string' ? block.name : '',
            args: '',
          });
        }
        break;
      }
      case 'content_block_delta': {
        const index = numberOrNull(data.index);
        if (index === null || !isRecord(data.delta)) break;
        const at = blocks.get(index);
        if (at === undefined) break;
        if (data.delta.type === 'text_delta' && typeof data.delta.text === 'string') {
          outText += data.delta.text;
        } else if (
          data.delta.type === 'input_json_delta' &&
          typeof data.delta.partial_json === 'string'
        ) {
          at.args += data.delta.partial_json;
        }
        break;
      }
      case 'message_delta': {
        if (isRecord(data.delta) && typeof data.delta.stop_reason === 'string') {
          stopFamily = anthropicFamily(data.delta.stop_reason);
        }
        if (isRecord(data.usage)) output = numberOrZero(data.usage.output_tokens);
        break;
      }
      default:
        break;
    }
  }
  const toolCalls: NormToolCall[] = [...blocks.entries()]
    .sort((a, b) => a[0] - b[0])
    .filter(([, b]) => b.kind === 'tool')
    .map(([, b]) => ({ id: b.id, name: b.name, args: b.args }));
  return { text: outText, toolCalls, stopFamily, usage: { input, output } };
}

/** Gemini's parts text: the text parts joined. */
function geminiPartsText(parts: unknown): string {
  if (!Array.isArray(parts)) return '';
  return parts
    .flatMap((p) => (isRecord(p) && typeof p.text === 'string' ? [p.text] : []))
    .join('');
}

/** One Gemini generateContent JSON answer as a normalized answer. */
export function parseGeminiJson(body: Json): NormAnswer {
  const candidates = Array.isArray(body.candidates) ? body.candidates : [];
  const candidate = isRecord(candidates[0]) ? candidates[0] : {};
  const content = isRecord(candidate.content) ? candidate.content : {};
  let outText = '';
  const toolCalls: NormToolCall[] = [];
  if (Array.isArray(content.parts)) {
    for (const part of content.parts) {
      if (!isRecord(part)) continue;
      if (typeof part.text === 'string') outText += part.text;
      else if (isRecord(part.functionCall)) {
        toolCalls.push({
          id: '',
          name: typeof part.functionCall.name === 'string' ? part.functionCall.name : '',
          args: JSON.stringify(part.functionCall.args ?? {}),
        });
      }
    }
  }
  const usage = isRecord(body.usageMetadata) ? body.usageMetadata : {};
  return {
    text: outText,
    toolCalls,
    stopFamily: geminiFamily(candidate.finishReason),
    usage: {
      input: numberOrZero(usage.promptTokenCount),
      output: numberOrZero(usage.candidatesTokenCount),
    },
  };
}

/** One Gemini streamGenerateContent SSE stream as a normalized answer. */
export function parseGeminiStream(text: string): NormAnswer {
  let outText = '';
  const toolCalls: NormToolCall[] = [];
  let stopFamily: StopFamily = 0;
  let input = 0;
  let output = 0;
  for (const frame of parseSseFrames(text)) {
    const chunk = parseJsonLine(frame.data);
    if (chunk === null) continue;
    const candidates = Array.isArray(chunk.candidates) ? chunk.candidates : [];
    const candidate = isRecord(candidates[0]) ? candidates[0] : {};
    const content = isRecord(candidate.content) ? candidate.content : {};
    if (Array.isArray(content.parts)) {
      for (const part of content.parts) {
        if (!isRecord(part)) continue;
        if (typeof part.text === 'string') outText += part.text;
        else if (isRecord(part.functionCall)) {
          toolCalls.push({
            id: '',
            name: typeof part.functionCall.name === 'string' ? part.functionCall.name : '',
            args: JSON.stringify(part.functionCall.args ?? {}),
          });
        }
      }
    }
    if (typeof candidate.finishReason === 'string') stopFamily = geminiFamily(candidate.finishReason);
    if (isRecord(chunk.usageMetadata)) {
      input = numberOrZero(chunk.usageMetadata.promptTokenCount);
      output = numberOrZero(chunk.usageMetadata.candidatesTokenCount);
    }
  }
  return { text: outText, toolCalls, stopFamily, usage: { input, output } };
}

// ---------------------------------------------------------------------------
// Rendering in the client's wire
// ---------------------------------------------------------------------------

function openAIFinish(family: StopFamily, hasToolCalls: boolean): string {
  if (family === 1) return 'length';
  if (family === 2 || hasToolCalls) return 'tool_calls';
  if (family === 3) return 'content_filter';
  return 'stop';
}

function anthropicStop(family: StopFamily, hasToolCalls: boolean): string {
  if (family === 1) return 'max_tokens';
  if (family === 2 || hasToolCalls) return 'tool_use';
  if (family === 3) return 'refusal';
  return 'end_turn';
}

/** A normalized answer as an OpenAI Chat Completions JSON object. */
export function renderOpenAIJson(answer: NormAnswer, model: string): Json {
  const message: Json = { role: 'assistant', content: answer.text };
  if (answer.toolCalls.length > 0) {
    message.tool_calls = answer.toolCalls.map((c) => ({
      id: c.id,
      type: 'function',
      function: { name: c.name, arguments: c.args },
    }));
  }
  return {
    id: 'chatcmpl-agg',
    object: 'chat.completion',
    created: FIXED_CREATED,
    model,
    choices: [
      {
        index: 0,
        message,
        finish_reason: openAIFinish(answer.stopFamily, answer.toolCalls.length > 0),
        logprobs: null,
      },
    ],
    usage: {
      prompt_tokens: answer.usage.input,
      completion_tokens: answer.usage.output,
      total_tokens: answer.usage.input + answer.usage.output,
    },
  };
}

/** One server-sent event frame. */
export function sseEvent(event: string | null, data: unknown): string {
  return `${event === null ? '' : `event: ${event}\n`}data: ${JSON.stringify(data)}\n\n`;
}

/** Split text into the short deltas a streamed answer is sent in. */
function splitText(text: string): string[] {
  const pieces: string[] = [];
  for (let at = 0; at < text.length; at += 13) pieces.push(text.slice(at, at + 13));
  return pieces.length === 0 ? [''] : pieces;
}

/** A normalized answer as an OpenAI SSE stream (ending with a usage chunk and [DONE]). */
export function renderOpenAIStream(answer: NormAnswer, model: string): string {
  const chunk = (delta: Json, finish: string | null): Json => ({
    id: 'chatcmpl-agg',
    object: 'chat.completion.chunk',
    created: FIXED_CREATED,
    model,
    choices: [{ index: 0, delta, finish_reason: finish }],
  });
  const frames: string[] = [sseEvent(null, chunk({ role: 'assistant', content: '' }, null))];
  for (const piece of splitText(answer.text)) {
    frames.push(sseEvent(null, chunk({ content: piece }, null)));
  }
  answer.toolCalls.forEach((call, index) => {
    frames.push(
      sseEvent(
        null,
        chunk(
          {
            tool_calls: [
              {
                index,
                id: call.id,
                type: 'function',
                function: { name: call.name, arguments: '' },
              },
            ],
          },
          null,
        ),
      ),
    );
    for (const piece of splitText(call.args)) {
      frames.push(
        sseEvent(
          null,
          chunk({ tool_calls: [{ index, function: { arguments: piece } }] }, null),
        ),
      );
    }
  });
  frames.push(
    sseEvent(null, chunk({}, openAIFinish(answer.stopFamily, answer.toolCalls.length > 0))),
  );
  frames.push(
    sseEvent(null, {
      id: 'chatcmpl-agg',
      object: 'chat.completion.chunk',
      created: FIXED_CREATED,
      model,
      choices: [],
      usage: {
        prompt_tokens: answer.usage.input,
        completion_tokens: answer.usage.output,
        total_tokens: answer.usage.input + answer.usage.output,
      },
    }),
  );
  frames.push('data: [DONE]\n\n');
  return frames.join('');
}

/** A normalized answer as an Anthropic Messages JSON object. */
export function renderAnthropicJson(answer: NormAnswer, model: string): Json {
  const content: Json[] = [];
  if (answer.text !== '') content.push({ type: 'text', text: answer.text });
  for (const call of answer.toolCalls) {
    content.push({
      type: 'tool_use',
      id: call.id !== '' ? call.id : 'toolu_agg',
      name: call.name,
      input: parseArgsObject(call.args),
    });
  }
  if (content.length === 0) content.push({ type: 'text', text: '' });
  return {
    id: 'msg_agg',
    type: 'message',
    role: 'assistant',
    model,
    content,
    stop_reason: anthropicStop(answer.stopFamily, answer.toolCalls.length > 0),
    stop_sequence: null,
    usage: { input_tokens: answer.usage.input, output_tokens: answer.usage.output },
  };
}

/** A normalized answer as an Anthropic SSE stream (message_start … message_stop). */
export function renderAnthropicStream(answer: NormAnswer, model: string): string {
  const frames: string[] = [
    sseEvent('message_start', {
      type: 'message_start',
      message: {
        id: 'msg_agg',
        type: 'message',
        role: 'assistant',
        model,
        content: [],
        stop_reason: null,
        stop_sequence: null,
        usage: { input_tokens: answer.usage.input, output_tokens: 0 },
      },
    }),
  ];
  let index = 0;
  if (answer.text !== '' || answer.toolCalls.length === 0) {
    const textIndex = index;
    index += 1;
    frames.push(
      sseEvent('content_block_start', {
        type: 'content_block_start',
        index: textIndex,
        content_block: { type: 'text', text: '' },
      }),
    );
    for (const piece of splitText(answer.text)) {
      frames.push(
        sseEvent('content_block_delta', {
          type: 'content_block_delta',
          index: textIndex,
          delta: { type: 'text_delta', text: piece },
        }),
      );
    }
    frames.push(sseEvent('content_block_stop', { type: 'content_block_stop', index: textIndex }));
  }
  for (const call of answer.toolCalls) {
    const callIndex = index;
    index += 1;
    frames.push(
      sseEvent('content_block_start', {
        type: 'content_block_start',
        index: callIndex,
        content_block: {
          type: 'tool_use',
          id: call.id !== '' ? call.id : `toolu_${callIndex}`,
          name: call.name,
          input: {},
        },
      }),
    );
    for (const piece of splitText(call.args)) {
      frames.push(
        sseEvent('content_block_delta', {
          type: 'content_block_delta',
          index: callIndex,
          delta: { type: 'input_json_delta', partial_json: piece },
        }),
      );
    }
    frames.push(sseEvent('content_block_stop', { type: 'content_block_stop', index: callIndex }));
  }
  frames.push(
    sseEvent('message_delta', {
      type: 'message_delta',
      delta: {
        stop_reason: anthropicStop(answer.stopFamily, answer.toolCalls.length > 0),
        stop_sequence: null,
      },
      usage: { output_tokens: answer.usage.output },
    }),
    sseEvent('message_stop', { type: 'message_stop' }),
  );
  return frames.join('');
}
