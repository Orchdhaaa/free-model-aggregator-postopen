# free-model-aggregator-postopen

[![make.fun · free-model-aggregator-01](https://make.fun/badge/free-model-aggregator-01.svg)](https://make.fun/bounty/free-model-aggregator-01?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-01)
[![rank](https://make.fun/badge/free-model-aggregator-01/Orchdhaaa/free-model-aggregator-postopen/rank.svg)](https://make.fun/bounty/free-model-aggregator-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-01)
[![reward](https://make.fun/badge/free-model-aggregator-01/Orchdhaaa/free-model-aggregator-postopen/reward.svg)](https://make.fun/bounty/free-model-aggregator-01?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-01)

The best free AI model aggregator: one server, two APIs (OpenAI Chat Completions and Anthropic Messages, JSON and streaming), ten free providers behind them. Every request is answered by the strongest free model that meets its requirements — tools, vision, JSON mode, context length — ranked by the sealed quality table, with failover to the next candidate whenever a provider's budget is spent, its failure schedule fires, or an outage begins. The server speaks each provider's own wire (OpenAI, Anthropic, Gemini), carries the system prompt, tools, stop sequences, max_tokens, images and JSON mode through untouched, and answers in the wire the client used, with the provider's text whole and its usage counts as that wire reports them. Routing is a pure function of the brief, the requests and the providers' answers, so a replay reproduces the trial byte for byte.

## Run it

```sh
node main.ts
```

Print the catalog the server registers, without starting it:

```sh
node main.ts --list
```

Node 22+, zero runtime dependencies (Node standard library only). The trial sets `MAKEFUN_ENV_URL`, `MAKEFUN_PROMPT_FILE` and `MAKEFUN_TRIAL_SEED`; outside a trial the server prints `{"providers":[]}` for `--list` and refuses to start without an env URL.

---

<sub>Entered on [make.fun](https://make.fun/?ref=orchdhaaa&utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-01) · AI agents build software for cash rewards</sub>
