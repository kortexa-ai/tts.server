#!/usr/bin/env node

const fs = require("node:fs");
const path = require("node:path");

const DEFAULTS = {
  url: process.env.TTS_URL || "http://127.0.0.1:4003",
  text:
    process.env.TTS_TEXT ||
    "Hello from the Kortexa smoke test. If you can hear this, the new OpenAI-style endpoint is alive.",
  instructions: process.env.TTS_INSTRUCTIONS || "",
  responseFormat: process.env.TTS_RESPONSE_FORMAT || "wav",
  output:
    process.env.TTS_OUTPUT ||
    path.join(process.cwd(), "tests", "output", "tts-smoke-test.wav"),
};

function printHelp() {
  console.log(`Usage: node tests/test.js [options]

Quick smoke test for the OpenAI-style TTS API.

Options:
  --url <url>              Base server URL (default: ${DEFAULTS.url})
  --text <text>            Text to synthesize
  --instructions <text>    Optional style instruction
  --voice <id>             Voice id from GET /v1/voices
  --format <format>        mp3 | wav | flac | pcm | aac | opus
  --stream                 Use stream_format=audio (PCM only)
  --out <path>             Output file path
  --list-voices            Print GET /v1/voices and exit
  --help                   Show this help
`);
}

function requireValue(argv, index, flag) {
  const value = argv[index];
  if (value === undefined || value.startsWith("--")) {
    throw new Error(`Missing value for ${flag}`);
  }
  return value;
}

function parseArgs(argv) {
  const options = {
    ...DEFAULTS,
    voice: process.env.TTS_VOICE || null,
    stream: false,
    listVoices: false,
    help: false,
  };

  for (let i = 0; i < argv.length; i += 1) {
    const arg = argv[i];
    switch (arg) {
      case "--url":
        options.url = requireValue(argv, ++i, arg);
        break;
      case "--text":
        options.text = requireValue(argv, ++i, arg);
        break;
      case "--instructions":
        options.instructions = requireValue(argv, ++i, arg);
        break;
      case "--voice":
        options.voice = requireValue(argv, ++i, arg);
        break;
      case "--format":
        options.responseFormat = requireValue(argv, ++i, arg);
        break;
      case "--stream":
        options.stream = true;
        break;
      case "--out":
        options.output = requireValue(argv, ++i, arg);
        break;
      case "--list-voices":
        options.listVoices = true;
        break;
      case "--help":
      case "-h":
        options.help = true;
        break;
      default:
        throw new Error(`Unknown argument: ${arg}`);
    }
  }

  return options;
}

async function fetchJson(url) {
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`${response.status} ${response.statusText}: ${await response.text()}`);
  }
  return response.json();
}

function defaultOutputPath(options) {
  if (options.output !== DEFAULTS.output) {
    return options.output;
  }
  const ext = options.stream ? "pcm" : options.responseFormat;
  return path.join(process.cwd(), "tests", "output", `tts-smoke-test.${ext}`);
}

async function main() {
  let options;
  try {
    options = parseArgs(process.argv.slice(2));
  } catch (error) {
    console.error(`Argument error: ${error.message}`);
    printHelp();
    process.exit(1);
  }

  if (options.help) {
    printHelp();
    return;
  }

  const baseUrl = options.url.replace(/\/+$/, "");
  const health = await fetchJson(`${baseUrl}/health`);
  const voices = await fetchJson(`${baseUrl}/v1/voices`);
  const models = await fetchJson(`${baseUrl}/v1/models`);

  if (!health.ready) {
    throw new Error(`Server not ready: ${health.load_error || "unknown error"}`);
  }

  if (options.listVoices) {
    console.log(JSON.stringify(voices, null, 2));
    return;
  }

  const modelId = models.data[0]?.id;
  if (!modelId) {
    throw new Error("No model ids returned by /v1/models");
  }

  const voiceId = options.voice || voices.default_voice || voices.data[0]?.id;
  if (!voiceId) {
    throw new Error("No voice ids returned by /v1/voices");
  }

  const outputPath = defaultOutputPath(options);
  fs.mkdirSync(path.dirname(outputPath), { recursive: true });

  const body = {
    model: modelId,
    input: options.text,
    voice: voiceId,
    instructions: options.instructions,
    response_format: options.stream ? "pcm" : options.responseFormat,
  };
  if (options.stream) {
    body.stream_format = "audio";
  }

  console.log(`Server: ${baseUrl}`);
  console.log(`Model: ${modelId}`);
  console.log(`Voice: ${voiceId}`);

  const response = await fetch(`${baseUrl}/v1/audio/speech`, {
    method: "POST",
    headers: {
      "content-type": "application/json",
    },
    body: JSON.stringify(body),
  });

  if (!response.ok) {
    throw new Error(`${response.status} ${response.statusText}: ${await response.text()}`);
  }

  const audio = Buffer.from(await response.arrayBuffer());
  fs.writeFileSync(outputPath, audio);

  console.log(`Saved ${audio.length} bytes to ${outputPath}`);
  console.log(`Content-Type: ${response.headers.get("content-type") || "(missing)"}`);
  console.log(`Sample rate: ${response.headers.get("x-sample-rate") || health.sample_rate}`);
}

main().catch((error) => {
  console.error(`Smoke test failed: ${error.message}`);
  process.exit(1);
});
