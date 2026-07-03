#!/usr/bin/env node
'use strict';
const fs = require('fs');

function main() {
  const inPath = process.argv[2];
  const outPath = process.argv[3];
  if (!inPath || !outPath) { console.error('usage: analyze.js <in> <out>'); process.exit(1); }
  const data = JSON.parse(fs.readFileSync(inPath, 'utf8'));
  const fileNodes = data.fileNodes || [];
  const importEdges = data.importEdges || [];
  const allEdges = data.allEdges || [];

  const byId = {}; fileNodes.forEach(n => byId[n.id] = n);

  // ---- common prefix of dir portions ----
  const paths = fileNodes.map(n => n.filePath);
  function dirOf(p){ const i = p.lastIndexOf('/'); return i < 0 ? '' : p.slice(0, i); }
  const dirs = paths.map(dirOf);
  // common prefix by path segment among directories (only when all share)
  function commonPrefixSegs(list){
    const split = list.map(d => d === '' ? [] : d.split('/'));
    if (!split.length) return [];
    let pref = split[0].slice();
    for (const s of split){
      let k=0; while(k<pref.length && k<s.length && pref[k]===s[k]) k++;
      pref = pref.slice(0,k);
      if (!pref.length) break;
    }
    return pref;
  }
  const prefSegs = commonPrefixSegs(dirs);
  const prefix = prefSegs.length ? prefSegs.join('/') + '/' : '';

  function groupOf(p){
    let rest = p;
    if (prefix && rest.startsWith(prefix)) rest = rest.slice(prefix.length);
    const seg = rest.split('/');
    if (seg.length <= 1) return '(root)';
    return seg[0];
  }

  // ---- A. directory groups ----
  const directoryGroups = {};
  const fileGroup = {};
  fileNodes.forEach(n => {
    const grp = groupOf(n.filePath);
    fileGroup[n.id] = grp;
    (directoryGroups[grp] = directoryGroups[grp] || []).push(n.id);
  });

  // ---- B. node type groups ----
  const nodeTypeGroups = {};
  fileNodes.forEach(n => (nodeTypeGroups[n.type] = nodeTypeGroups[n.type] || []).push(n.id));

  // ---- C. fan in/out (imports) ----
  const fanOut = {}, fanIn = {};
  fileNodes.forEach(n => { fanOut[n.id]=0; fanIn[n.id]=0; });
  importEdges.forEach(e => { if(fanOut[e.source]!=null) fanOut[e.source]++; if(fanIn[e.target]!=null) fanIn[e.target]++; });

  // ---- D. cross-category edges ----
  const ccMap = {};
  allEdges.forEach(e => {
    const st = byId[e.source]?.type, tt = byId[e.target]?.type;
    if (!st || !tt) return;
    if (st === tt) return; // only cross-type
    const key = st+'>'+tt+'>'+e.type;
    ccMap[key] = (ccMap[key]||0)+1;
  });
  const crossCategoryEdges = Object.entries(ccMap).map(([k,c])=>{const[ft,tt,et]=k.split('>');return{fromType:ft,toType:tt,edgeType:et,count:c};});

  // ---- E. inter-group imports (use all relational edges treated as dependency) ----
  // Use importEdges + depends_on/configures/tested_by for direction signals among groups
  const depEdges = allEdges.filter(e => ['imports','depends_on','calls','configures','tested_by'].includes(e.type));
  const interMap = {};
  depEdges.forEach(e => {
    const a = fileGroup[e.source], b = fileGroup[e.target];
    if (a==null||b==null||a===b) return;
    const key = a+'>'+b; interMap[key]=(interMap[key]||0)+1;
  });
  const interGroupImports = Object.entries(interMap).map(([k,c])=>{const[from,to]=k.split('>');return{from,to,count:c};});

  // ---- F. intra-group density ----
  const intraGroupDensity = {};
  Object.keys(directoryGroups).forEach(grp => {
    let internal=0, total=0;
    depEdges.forEach(e => {
      const a=fileGroup[e.source], b=fileGroup[e.target];
      if(a===grp||b===grp){ total++; if(a===grp&&b===grp) internal++; }
    });
    intraGroupDensity[grp] = { internalEdges:internal, totalEdges:total, density: total?+(internal/total).toFixed(3):0 };
  });

  // ---- G. pattern matching ----
  const dirPat = [
    [/^(routes|api|controllers|endpoints|handlers)$/,'api'],
    [/^(services|core|lib|domain|logic)$/,'service'],
    [/^(models|db|data|persistence|repository|entities)$/,'data'],
    [/^(components|views|pages|ui|layouts|screens)$/,'ui'],
    [/^(middleware|plugins|interceptors|guards)$/,'middleware'],
    [/^(utils|helpers|common|shared|tools)$/,'utility'],
    [/^(config|constants|env|settings)$/,'config'],
    [/^(__tests__|test|tests|spec|specs)$/,'test'],
    [/^(types|interfaces|schemas|contracts|dtos)$/,'types'],
    [/^hooks$/,'hooks'],
    [/^(store|state|reducers|actions|slices)$/,'state'],
    [/^(assets|static|public)$/,'assets'],
    [/^migrations$/,'data'],
    [/^(scripts|bin)$/,'tooling'],
    [/^(benchmarks?)$/,'test'],
    [/^(capabilities|docs|documentation|wiki)$/,'documentation'],
    [/^(deploy|deployment|infra|infrastructure|docker)$/,'infrastructure'],
  ];
  function matchDir(name){
    for(const[re,lab] of dirPat) if(re.test(name)) return lab;
    return null;
  }
  function matchFile(n){
    const fp = n.filePath, base = n.name;
    if(/\.(test|spec)\.[^.]+$/.test(base)||/^test_.*\.py$/.test(base)) return 'test';
    if(/\.d\.ts$/.test(base)) return 'types';
    if(/\.(md|rst)$/.test(base)) return 'documentation';
    if(/\.(sql)$/.test(base)) return 'data';
    if(/\.(graphql|gql|proto)$/.test(base)) return 'types';
    if(/\.fs$/.test(base)) return 'reference';
    if(base==='__init__.py') return 'entry';
    return null;
  }
  const patternMatches = {};
  Object.keys(directoryGroups).forEach(g => { const m = matchDir(g); if(m) patternMatches[g]=m; });
  const filePatternMatches = {};
  fileNodes.forEach(n => { const m=matchFile(n); if(m) filePatternMatches[n.id]=m; });

  // ---- H. deployment topology ----
  const names = fileNodes.map(n=>n.filePath);
  const has = (re)=>names.some(p=>re.test(p));
  const infraFiles = names.filter(p=>/Dockerfile|docker-compose|\.tf$|k8s|kubernetes|helm|\.github\/workflows|gitlab-ci|Jenkinsfile|Makefile/.test(p));
  const deploymentTopology = {
    hasDockerfile: has(/Dockerfile/), hasCompose: has(/docker-compose/), hasK8s: has(/k8s|kubernetes|helm/),
    hasTerraform: has(/\.tf$/), hasCI: has(/\.github\/workflows|gitlab-ci|Jenkinsfile/), infraFiles
  };

  // ---- I. data pipeline ----
  const dataPipeline = {
    schemaFiles: names.filter(p=>/\.(sql|graphql|proto|prisma)$/.test(p)),
    migrationFiles: names.filter(p=>/migrations\//.test(p)),
    dataModelFiles: names.filter(p=>/(models|entities)\//.test(p)),
    apiHandlerFiles: names.filter(p=>/(routes|controllers|api)\//.test(p)),
  };

  // ---- J. doc coverage ----
  const docFiles = fileNodes.filter(n=>n.type==='document'||/\.(md|rst)$/.test(n.name));
  const groups = Object.keys(directoryGroups);
  const groupsWithDocs = groups.filter(g => directoryGroups[g].some(id=>docFiles.find(d=>d.id===id))).length;
  const undocumentedGroups = groups.filter(g => !directoryGroups[g].some(id=>docFiles.find(d=>d.id===id)));
  const docCoverage = { groupsWithDocs, totalGroups: groups.length, coverageRatio: +(groupsWithDocs/groups.length).toFixed(2), undocumentedGroups };

  // ---- K. dependency direction ----
  const dependencyDirection = [];
  const seen = new Set();
  interGroupImports.forEach(({from,to})=>{
    const fwd = interMap[from+'>'+to]||0, rev = interMap[to+'>'+from]||0;
    const k=[from,to].sort().join('|');
    if(seen.has(k)) return; seen.add(k);
    if(fwd>=rev && fwd>0) dependencyDirection.push({dependent:from,dependsOn:to});
    else if(rev>fwd) dependencyDirection.push({dependent:to,dependsOn:from});
  });

  const filesPerGroup={}; Object.keys(directoryGroups).forEach(g=>filesPerGroup[g]=directoryGroups[g].length);
  const nodeTypeCounts={}; Object.keys(nodeTypeGroups).forEach(t=>nodeTypeCounts[t]=nodeTypeGroups[t].length);

  const result = {
    scriptCompleted:true, commonPrefix:prefix, directoryGroups, nodeTypeGroups,
    crossCategoryEdges, interGroupImports, intraGroupDensity, patternMatches, filePatternMatches,
    deploymentTopology, dataPipeline, docCoverage, dependencyDirection,
    fileStats:{ totalFileNodes:fileNodes.length, filesPerGroup, nodeTypeCounts },
    fileFanIn:fanIn, fileFanOut:fanOut
  };
  fs.writeFileSync(outPath, JSON.stringify(result,null,2));
  process.exit(0);
}
try{ main(); } catch(e){ console.error(e.stack||String(e)); process.exit(1); }
