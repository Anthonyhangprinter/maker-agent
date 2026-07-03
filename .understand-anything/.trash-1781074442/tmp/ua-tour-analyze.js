#!/usr/bin/env node
"use strict";
const fs = require("fs");

function main() {
  const inPath = process.argv[2];
  const outPath = process.argv[3];
  if (!inPath || !outPath) {
    console.error("usage: ua-tour-analyze.js <input.json> <output.json>");
    process.exit(1);
  }
  const data = JSON.parse(fs.readFileSync(inPath, "utf8"));
  const nodes = data.nodes || [];
  const edges = data.edges || [];
  const layers = data.layers || [];

  const nodeById = new Map();
  for (const n of nodes) nodeById.set(n.id, n);

  // Adjacency
  const outAdj = new Map(); // source -> [{target,type}]
  const inAdj = new Map();
  for (const n of nodes) { outAdj.set(n.id, []); inAdj.set(n.id, []); }
  for (const e of edges) {
    if (!outAdj.has(e.source)) outAdj.set(e.source, []);
    if (!inAdj.has(e.target)) inAdj.set(e.target, []);
    outAdj.get(e.source).push({ target: e.target, type: e.type });
    inAdj.get(e.target).push({ source: e.source, type: e.type });
  }

  // Fan-in / fan-out (distinct neighbors)
  function distinctTargets(list, key) {
    const s = new Set();
    for (const x of list) s.add(x[key]);
    return s.size;
  }
  const fanIn = {}, fanOut = {};
  for (const n of nodes) {
    fanIn[n.id] = distinctTargets(inAdj.get(n.id) || [], "source");
    fanOut[n.id] = distinctTargets(outAdj.get(n.id) || [], "target");
  }

  const fanInRanking = nodes.map(n => ({ id: n.id, fanIn: fanIn[n.id], name: n.name }))
    .sort((a, b) => b.fanIn - a.fanIn).slice(0, 20);
  const fanOutRanking = nodes.map(n => ({ id: n.id, fanOut: fanOut[n.id], name: n.name }))
    .sort((a, b) => b.fanOut - a.fanOut).slice(0, 20);

  // Entry point candidates
  const entryNames = new Set(["index.ts","index.js","main.ts","main.js","app.ts","app.js","server.ts","server.js","mod.rs","main.go","main.py","main.rs","manage.py","app.py","wsgi.py","asgi.py","run.py","__main__.py","Application.java","Main.java","Program.cs","config.ru","index.php","App.swift","Application.kt","main.cpp","main.c"]);
  const fanOutVals = nodes.map(n => fanOut[n.id]).sort((a, b) => a - b);
  const fanInVals = nodes.map(n => fanIn[n.id]).sort((a, b) => a - b);
  const p90FanOut = fanOutVals[Math.floor(0.9 * (fanOutVals.length - 1))] || 0;
  const p25FanIn = fanInVals[Math.floor(0.25 * (fanInVals.length - 1))] || 0;

  const entryScores = [];
  for (const n of nodes) {
    let score = 0;
    const fp = n.filePath || "";
    if (n.type === "document") {
      if (n.name === "README.md" && !fp.includes("/")) score += 5;
      else if (/\.md$/i.test(n.name) && !fp.includes("/")) score += 2;
    } else if (n.type === "file") {
      if (entryNames.has(n.name)) score += 3;
      const depth = fp.split("/").length;
      if (depth <= 2) score += 1;
      if (fanOut[n.id] >= p90FanOut && p90FanOut > 0) score += 1;
      if (fanIn[n.id] <= p25FanIn) score += 1;
    }
    if (score > 0) entryScores.push({ id: n.id, score, name: n.name, summary: n.summary || "" });
  }
  entryScores.sort((a, b) => b.score - a.score);
  const entryPointCandidates = entryScores.slice(0, 5);

  // BFS from top CODE entry point
  let startNode = null;
  for (const c of entryScores) {
    const nn = nodeById.get(c.id);
    if (nn && nn.type !== "document") { startNode = c.id; break; }
  }
  if (!startNode) {
    // fallback: highest fanOut file
    const f = fanOutRanking.find(r => { const nn = nodeById.get(r.id); return nn && nn.type === "file"; });
    startNode = f ? f.id : (nodes[0] && nodes[0].id);
  }
  const traverseTypes = new Set(["imports", "calls"]);
  const order = [];
  const depthMap = {};
  if (startNode) {
    const q = [startNode];
    depthMap[startNode] = 0;
    while (q.length) {
      const cur = q.shift();
      order.push(cur);
      for (const e of outAdj.get(cur) || []) {
        if (!traverseTypes.has(e.type)) continue;
        if (depthMap[e.target] === undefined) {
          depthMap[e.target] = depthMap[cur] + 1;
          q.push(e.target);
        }
      }
    }
  }
  const byDepth = {};
  for (const id of order) {
    const d = depthMap[id];
    (byDepth[d] = byDepth[d] || []).push(id);
  }

  // Non-code inventory
  const nonCodeFiles = { documentation: [], infrastructure: [], data: [], config: [] };
  for (const n of nodes) {
    const rec = { id: n.id, name: n.name, type: n.type, summary: n.summary || "" };
    if (n.type === "document") nonCodeFiles.documentation.push(rec);
    else if (["service", "pipeline", "resource"].includes(n.type)) nonCodeFiles.infrastructure.push(rec);
    else if (["table", "schema", "endpoint"].includes(n.type)) nonCodeFiles.data.push(rec);
    else if (n.type === "config") nonCodeFiles.config.push(rec);
  }

  // Clusters: bidirectional imports/calls pairs, then expand
  const pairKey = (a, b) => [a, b].sort().join("||");
  const directed = new Set();
  for (const e of edges) {
    if (e.type === "imports" || e.type === "calls") directed.add(e.source + ">>" + e.target);
  }
  const biPairs = [];
  const seenPair = new Set();
  for (const e of edges) {
    if (e.type !== "imports" && e.type !== "calls") continue;
    if (directed.has(e.target + ">>" + e.source)) {
      const k = pairKey(e.source, e.target);
      if (!seenPair.has(k)) { seenPair.add(k); biPairs.push([e.source, e.target]); }
    }
  }
  // Edge counts between node sets
  function edgesBetween(setIds) {
    let c = 0;
    for (const e of edges) {
      if (setIds.has(e.source) && setIds.has(e.target)) c++;
    }
    return c;
  }
  const clusters = [];
  for (const [a, b] of biPairs) {
    const cl = new Set([a, b]);
    // expand: add nodes connecting to 2+ members
    for (const n of nodes) {
      if (cl.has(n.id)) continue;
      let conn = 0;
      for (const e of edges) {
        if (e.type !== "imports" && e.type !== "calls") continue;
        if ((e.source === n.id && cl.has(e.target)) || (e.target === n.id && cl.has(e.source))) conn++;
      }
      if (conn >= 2 && cl.size < 5) cl.add(n.id);
    }
    clusters.push({ nodes: Array.from(cl), edgeCount: edgesBetween(cl) });
  }
  clusters.sort((a, b) => b.edgeCount - a.edgeCount);
  const topClusters = clusters.slice(0, 10);

  // Layers
  const layerOut = {
    count: layers.length,
    list: layers.map(l => ({ id: l.id, name: l.name, description: l.description }))
  };

  // Node summary index
  const nodeSummaryIndex = {};
  for (const n of nodes) {
    nodeSummaryIndex[n.id] = { name: n.name, type: n.type, summary: n.summary || "" };
  }

  const result = {
    scriptCompleted: true,
    entryPointCandidates,
    fanInRanking,
    fanOutRanking,
    bfsTraversal: { startNode, order, depthMap, byDepth },
    nonCodeFiles,
    clusters: topClusters,
    layers: layerOut,
    nodeSummaryIndex,
    totalNodes: nodes.length,
    totalEdges: edges.length
  };
  fs.writeFileSync(outPath, JSON.stringify(result, null, 2));
  process.exit(0);
}

try { main(); } catch (e) { console.error(e && e.stack ? e.stack : String(e)); process.exit(1); }
