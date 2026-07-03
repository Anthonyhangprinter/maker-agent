const fs = require('fs');
const g = JSON.parse(fs.readFileSync('/home/theultimatecunt/.openclaw/skills/cad-builder/.understand-anything/intermediate/assembled-graph.json', 'utf8'));

const isFileLevel = (id) => !id.startsWith('function:');
const fileNodes = g.nodes.filter(n => isFileLevel(n.id)).map(n => ({
  id: n.id, type: n.type, name: n.name, filePath: n.filePath, summary: n.summary, tags: n.tags || []
}));
const idset = new Set(fileNodes.map(n => n.id));

// file-level edges only
const allEdges = g.edges.filter(e => idset.has(e.source) && idset.has(e.target));
const importEdges = allEdges.filter(e => e.type === 'imports');

const out = { fileNodes, importEdges, allEdges };
fs.writeFileSync('/home/theultimatecunt/.openclaw/skills/cad-builder/.understand-anything/tmp/ua-arch-input.json', JSON.stringify(out, null, 2));
console.log('fileNodes:', fileNodes.length, 'allEdges:', allEdges.length, 'importEdges:', importEdges.length);
