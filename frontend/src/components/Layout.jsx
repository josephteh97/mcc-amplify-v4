import React, { useState, useEffect, useCallback, useMemo } from 'react';
import { useGLTF } from '@react-three/drei';
import UploadPanel from './UploadPanel';
import ChatPanel from './ChatPanel';
import Viewer from './Viewer';
import EditPanel from './EditPanel';

const Layout = () => {
  const [jobId, setJobId]       = useState(null);
  const [modelUrl, setModelUrl] = useState(null);
  const [rvtUrl, setRvtUrl]     = useState(null);
  const [fileName, setFileName] = useState('');
  const [modelReady, setModelReady] = useState(false);
  const [modelStats, setModelStats] = useState(null);

  // ── Human-in-the-loop correction state ──────────────────────────────────────
  const [recipe, setRecipe]                   = useState(null);
  const [selectedElement, setSelectedElement] = useState(null); // {type, pluralType, index, data}
  const [isPatching, setIsPatching]           = useState(false);
  const [isRebuilding, setIsRebuilding]       = useState(false);
  const [elementDefaults, setElementDefaults] = useState({});

  // ── Confidence gate — shown before a risky Revit commit ─────────────────────
  const [revitGate, setRevitGate]             = useState(null); // null | { reasons: string[] }

  // ── Project profile ──────────────────────────────────────────────────────────
  const [profile, setProfile]                 = useState(null);
  const [profileOpen, setProfileOpen]         = useState(false);
  const [profileSaving, setProfileSaving]     = useState(false);
  const [profileDraft, setProfileDraft]       = useState({});

  const handleJobCreated = (id) => setJobId(id);

  const handleProcessingComplete = (id, result, name) => {
    setJobId(id);
    setFileName(name || 'floor_plan.pdf');
    if (result.files?.gltf) setModelUrl(`/api/download/gltf/${id}`);
    if (result.files?.rvt)  setRvtUrl(`/api/download/rvt/${id}`);
    setModelStats(result.stats || null);
    setModelReady(true);
  };

  const handleReset = () => {
    setJobId(null);
    setModelUrl(null);
    setRvtUrl(null);
    setFileName('');
    setModelReady(false);
    setRecipe(null);
    setSelectedElement(null);
    setModelStats(null);
    setRevitGate(null);
  };

  // Fetch project profile on mount
  useEffect(() => {
    fetch('/api/project_profile').then(r => r.json()).then(p => {
      setProfile(p);
      setProfileDraft(p);
    }).catch(() => {});
  }, []);

  const handleSaveProfile = async () => {
    setProfileSaving(true);
    try {
      await fetch('/api/project_profile', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(profileDraft),
      });
      setProfile(profileDraft);
      setProfileOpen(false);
    } finally {
      setProfileSaving(false);
    }
  };

  // Fetch recipe once the model is ready so EditPanel can show element data
  useEffect(() => {
    if (!modelReady || !jobId) return;
    fetch(`/api/model/${jobId}/recipe`)
      .then(r => r.json())
      .then(setRecipe)
      .catch(err => console.warn('Recipe fetch failed:', err));
  }, [modelReady, jobId]);

  // Called by Viewer when user clicks a mesh — "wall" + 3 → recipe["walls"][3]
  const handleElementSelect = useCallback((type, index) => {
    if (!recipe) return;
    const pluralType = type + 's';   // "wall" → "walls", "column" → "columns", etc.
    const data = (recipe[pluralType] || [])[index];
    if (!data) return;
    setSelectedElement({ type, pluralType, index, data });
    // Fetch firm defaults for this element type from the corrections history
    fetch(`/api/corrections/defaults/${pluralType}`)
      .then(r => r.json())
      .then(setElementDefaults)
      .catch(() => setElementDefaults({}));
  }, [recipe]);

  // Elements where YOLO confidence < 0.6 — sorted lowest first
  const lowConfidenceItems = useMemo(() => {
    if (!recipe) return [];
    const items = [];
    ['walls', 'columns', 'doors', 'windows', 'floors', 'ceilings'].forEach(pluralType => {
      (recipe[pluralType] || []).forEach((el, idx) => {
        if (el.confidence !== undefined && el.confidence < 0.6) {
          items.push({
            type:       pluralType.slice(0, -1),  // "columns" → "column"
            pluralType,
            index:      idx,
            confidence: el.confidence,
          });
        }
      });
    });
    return items.sort((a, b) => a.confidence - b.confidence);
  }, [recipe]);

  // Apply a patch to a single element and immediately rebuild glTF
  const handlePatch = useCallback(async (changes, del = false) => {
    if (!selectedElement || !jobId) return;
    setIsPatching(true);
    try {
      const res = await fetch(`/api/model/${jobId}/recipe`, {
        method:  'PATCH',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          element_type:  selectedElement.pluralType,
          element_index: selectedElement.index,
          changes,
          delete: del,
        }),
      });
      if (!res.ok) {
        const msg = await res.text();
        throw new Error(msg);
      }

      // Cache-bust so useGLTF fetches the freshly-exported GLB
      useGLTF.clear(`/api/download/gltf/${jobId}`);
      setModelUrl(`/api/download/gltf/${jobId}?t=${Date.now()}`);

      // Re-fetch recipe and sync selectedElement with the updated data
      const newRecipe = await fetch(`/api/model/${jobId}/recipe`).then(r => r.json());
      setRecipe(newRecipe);

      const updated = newRecipe[selectedElement.pluralType]?.[selectedElement.index];
      if (updated) {
        setSelectedElement(prev => ({ ...prev, data: updated }));
      } else {
        // Element was deleted — deselect
        setSelectedElement(null);
      }
    } catch (err) {
      console.error('Patch failed:', err);
      alert(`Correction failed: ${err.message}`);
    } finally {
      setIsPatching(false);
    }
  }, [selectedElement, jobId]);

  // Trigger full RVT rebuild — after confidence gate passes
  const _doRebuildRevit = useCallback(async () => {
    if (!jobId) return;
    setRevitGate(null);
    setIsRebuilding(true);
    try {
      const res = await fetch(`/api/rebuild/${jobId}`, { method: 'POST' });
      if (!res.ok) throw new Error(await res.text());

      const poll = setInterval(async () => {
        const st = await fetch(`/api/status/${jobId}`).then(r => r.json());
        if (st.status === 'completed' || st.status === 'failed') {
          clearInterval(poll);
          setIsRebuilding(false);
          if (st.result?.files?.rvt) setRvtUrl(`/api/download/rvt/${jobId}`);
        }
      }, 2000);
    } catch (err) {
      console.error('Rebuild failed:', err);
      setIsRebuilding(false);
      alert(`Revit rebuild failed: ${err.message}`);
    }
  }, [jobId]);

  const handleRebuildRevit = useCallback(() => {
    if (!jobId) return;
    // ── Confidence gate: warn before committing a low-quality model to Revit ──
    const reasons = [];
    if (modelStats?.grid_confidence_label === 'Fallback')
      reasons.push('Grid detection failed — element coordinates may be unreliable.');
    if (modelStats?.is_scanned)
      reasons.push('Scanned PDF detected — vector data unavailable, accuracy reduced.');
    const totalElements = recipe
      ? Object.values(recipe).filter(Array.isArray).reduce((s, a) => s + a.length, 0)
      : 0;
    if (totalElements > 0 && lowConfidenceItems.length / totalElements > 0.3)
      reasons.push(`${lowConfidenceItems.length} of ${totalElements} elements are low-confidence — consider reviewing first.`);

    if (reasons.length > 0) {
      setRevitGate({ reasons });
      return;
    }
    _doRebuildRevit();
  }, [jobId, modelStats, recipe, lowConfidenceItems, _doRebuildRevit]);

  return (
    <div className="w-screen h-screen flex flex-col bg-slate-950 overflow-hidden">

      {/* ── Top header bar ─────────────────────────────────────────── */}
      <header className="h-12 shrink-0 flex items-center justify-between px-5 bg-slate-900 border-b border-slate-800 z-10">
        <div className="flex items-center gap-2.5">
          <span className="text-xl">🏗️</span>
          <span className="text-sm font-bold text-white tracking-wide">Amplify AI</span>
          <span className="text-slate-600 text-xs font-light">· Floor Plan → 3D BIM</span>
        </div>
        <span className="text-xs text-slate-500 font-medium">MCC Construction</span>
      </header>

      {/* ── Three-column workspace ──────────────────────────────────── */}
      <div className="flex flex-1 min-h-0">

        {/* ── LEFT: Upload + status + downloads ──────────────────── */}
        <aside className="w-72 shrink-0 flex flex-col bg-slate-900 border-r border-slate-800 overflow-y-auto">

          {/* Panel heading */}
          <div className="px-4 pt-4 pb-2">
            <p className="text-[11px] font-semibold text-slate-400 uppercase tracking-widest">Upload</p>
          </div>

          <div className="px-4 pb-4">
            <UploadPanel
              onJobCreated={handleJobCreated}
              onProcessingComplete={handleProcessingComplete}
            />
          </div>

          {/* Model stats — shown when model is ready */}
          {modelReady && modelStats && (
            <div className="px-4 pb-4 space-y-2">
              <p className="text-[11px] font-semibold text-slate-400 uppercase tracking-widest mb-2">Analysis</p>

              {/* Scanned PDF warning */}
              {modelStats.is_scanned && (
                <div className="flex items-start gap-2 bg-amber-900/40 border border-amber-700/60 rounded-lg px-3 py-2">
                  <span className="text-amber-400 text-sm shrink-0 mt-0.5">⚠</span>
                  <p className="text-[10px] text-amber-300 leading-relaxed">
                    Scanned PDF detected — grid uses fallback coordinates. Accuracy reduced.
                  </p>
                </div>
              )}

              {/* Grid confidence badge */}
              <div className="flex items-center justify-between bg-slate-800 rounded-lg px-3 py-2">
                <span className="text-[10px] text-slate-400">Grid confidence</span>
                <span className={`text-[10px] font-bold px-2 py-0.5 rounded-full ${
                  modelStats.grid_confidence_label === 'High'     ? 'bg-emerald-900/60 text-emerald-300' :
                  modelStats.grid_confidence_label === 'Medium'   ? 'bg-yellow-900/60 text-yellow-300'  :
                  modelStats.grid_confidence_label === 'Fallback' ? 'bg-red-900/60 text-red-300'        :
                                                                    'bg-slate-700 text-slate-400'
                }`}>
                  {modelStats.grid_confidence_label} ({(modelStats.grid_confidence * 100).toFixed(0)}%)
                </span>
              </div>

              {/* Element counts */}
              <div className="bg-slate-800 rounded-lg px-3 py-2 space-y-1">
                <div className="flex justify-between">
                  <span className="text-[10px] text-slate-400">Elements detected</span>
                  <span className="text-[10px] font-semibold text-slate-200">{modelStats.element_count}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-[10px] text-slate-400">Grid lines</span>
                  <span className="text-[10px] font-semibold text-slate-200">{modelStats.grid_lines}</span>
                </div>
              </div>

              {/* Pre-clash validation warnings */}
              {modelStats.validation_warnings?.length > 0 && (
                <div className="bg-red-900/20 border border-red-800/50 rounded-lg px-3 py-2 space-y-1.5">
                  <p className="text-[10px] font-semibold text-red-400 uppercase tracking-widest">
                    {modelStats.validation_warnings.length} Clash Warning{modelStats.validation_warnings.length > 1 ? 's' : ''}
                  </p>
                  {modelStats.validation_warnings.map((w, i) => (
                    <p key={i} className="text-[9px] text-red-300 leading-relaxed">• {w}</p>
                  ))}
                </div>
              )}
            </div>
          )}

          {/* Low-confidence elements — clickable to select for review */}
          {modelReady && lowConfidenceItems.length > 0 && (
            <div className="px-4 pb-4 space-y-2">
              <p className="text-[11px] font-semibold text-amber-500 uppercase tracking-widest mb-2">
                ⚠ Needs Review ({lowConfidenceItems.length})
              </p>
              <div className="space-y-1">
                {lowConfidenceItems.map((item, i) => (
                  <button
                    key={i}
                    onClick={() => handleElementSelect(item.type, item.index)}
                    className="w-full flex items-center justify-between bg-amber-900/20 border border-amber-800/40 hover:border-amber-600/60 hover:bg-amber-900/30 rounded-lg px-3 py-1.5 transition-colors text-left"
                  >
                    <span className="text-[10px] text-amber-200 capitalize">
                      {item.type} #{item.index}
                    </span>
                    <span className="text-[9px] font-mono text-amber-400">
                      {(item.confidence * 100).toFixed(0)}% conf
                    </span>
                  </button>
                ))}
              </div>
            </div>
          )}

          {/* Downloads — shown only when model is ready */}
          {modelReady && (
            <div className="px-4 pb-4 space-y-2">
              <p className="text-[11px] font-semibold text-slate-400 uppercase tracking-widest mb-3">Downloads</p>

              {rvtUrl && (
                <a
                  href={rvtUrl}
                  download
                  className="flex items-center gap-2 w-full bg-indigo-600 hover:bg-indigo-500 text-white text-xs font-semibold px-4 py-2.5 rounded-lg transition-colors"
                >
                  <span>⬇</span> Download RVT
                </a>
              )}
              {modelUrl && (
                <a
                  href={modelUrl}
                  download
                  className="flex items-center gap-2 w-full bg-blue-600 hover:bg-blue-500 text-white text-xs font-semibold px-4 py-2.5 rounded-lg transition-colors"
                >
                  <span>⬇</span> Download glTF
                </a>
              )}
              <button
                onClick={handleReset}
                className="flex items-center gap-2 w-full bg-slate-700 hover:bg-slate-600 text-slate-200 text-xs font-semibold px-4 py-2.5 rounded-lg transition-colors"
              >
                ↺ New Upload
              </button>
            </div>
          )}

          {/* Project settings button */}
          <div className="px-4 pb-3">
            <button
              onClick={() => setProfileOpen(true)}
              className="w-full flex items-center gap-2 bg-slate-800 hover:bg-slate-700 border border-slate-700 text-slate-300 text-xs font-medium px-3 py-2 rounded-lg transition-colors"
            >
              <span>⚙</span>
              <span>Project Profile</span>
              {profile?.building_type && (
                <span className="ml-auto text-[9px] text-slate-500 capitalize">{profile.building_type}</span>
              )}
            </button>
          </div>

          {/* Feature chips */}
          <div className="mt-auto px-4 pb-5 pt-3 border-t border-slate-800 grid grid-cols-2 gap-2">
            {[
              { icon: '🧠', label: 'YOLO + AI' },
              { icon: '📐', label: 'Dual-Track' },
              { icon: '🛡️', label: 'DoS-Safe' },
              { icon: '📦', label: 'Native RVT' },
            ].map((f) => (
              <div key={f.label} className="flex items-center gap-1.5 bg-slate-800 rounded-lg px-2 py-1.5">
                <span className="text-sm">{f.icon}</span>
                <span className="text-[10px] text-slate-400 font-medium">{f.label}</span>
              </div>
            ))}
          </div>
        </aside>

        {/* ── CENTER: 3D Viewer + EditPanel overlay ───────────────── */}
        <main className="flex-1 relative bg-slate-950 min-w-0">
          {/* Status badge when model is ready */}
          {modelReady && (
            <div className="absolute top-3 left-3 z-10">
              <div className="flex items-center gap-2 bg-slate-900/80 backdrop-blur-sm border border-slate-700 rounded-xl px-3 py-1.5">
                <span className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse"></span>
                <span className="text-xs font-semibold text-slate-200 max-w-[180px] truncate">{fileName}</span>
              </div>
            </div>
          )}

          {/* Click hint — shown when model is ready and nothing is selected */}
          {modelReady && !selectedElement && (
            <div className="absolute top-3 right-3 z-10">
              <div className="bg-slate-900/70 backdrop-blur-sm border border-slate-800 text-slate-400 rounded-lg px-3 py-1.5 text-[10px]">
                Click any element to edit
              </div>
            </div>
          )}

          {/* Controls hint */}
          <div className="absolute bottom-3 left-3 z-10">
            <div className="bg-slate-900/70 backdrop-blur-sm border border-slate-800 text-slate-400 rounded-lg px-3 py-2 text-[10px] space-y-0.5">
              <p><span className="text-slate-200 font-semibold">Rotate</span> · left drag</p>
              <p><span className="text-slate-200 font-semibold">Pan</span> · right drag</p>
              <p><span className="text-slate-200 font-semibold">Zoom</span> · scroll</p>
            </div>
          </div>

          <Viewer
            modelUrl={modelUrl}
            onElementSelect={handleElementSelect}
            selectedMesh={selectedElement}
          />

          {/* EditPanel — absolute overlay, top-right of center column */}
          {selectedElement && (
            <div className="absolute top-3 right-3 z-20 w-72">
              <EditPanel
                element={selectedElement.data}
                elementType={selectedElement.type}
                elementIndex={selectedElement.index}
                elementDefaults={elementDefaults}
                isPatching={isPatching}
                isRebuilding={isRebuilding}
                onApply={handlePatch}
                onDelete={() => handlePatch({}, true)}
                onClose={() => setSelectedElement(null)}
                onSendToRevit={handleRebuildRevit}
              />
            </div>
          )}
        </main>

        {/* ── RIGHT: Chat agent ───────────────────────────────────── */}
        <aside className="w-80 shrink-0 flex flex-col bg-slate-900 border-l border-slate-800">
          {/* Panel heading */}
          <div className="px-4 pt-4 pb-2 shrink-0">
            <p className="text-[11px] font-semibold text-slate-400 uppercase tracking-widest">AI Supervisor</p>
          </div>
          <div className="flex-1 min-h-0 px-3 pb-3">
            <div className="h-full">
              <ChatPanel jobId={jobId} />
            </div>
          </div>
        </aside>

      </div>

      {/* ── Confidence gate modal ──────────────────────────────────────────── */}
      {revitGate && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-sm">
          <div className="bg-slate-900 border border-amber-700/60 rounded-2xl shadow-2xl w-[400px] p-6">
            <p className="text-amber-400 text-sm font-bold mb-1">Quality Warning</p>
            <p className="text-slate-300 text-xs mb-4 leading-relaxed">
              The model may have reliability issues. Review before committing to Revit:
            </p>
            <ul className="space-y-2 mb-5">
              {revitGate.reasons.map((r, i) => (
                <li key={i} className="flex items-start gap-2 bg-amber-900/20 border border-amber-800/40 rounded-lg px-3 py-2">
                  <span className="text-amber-400 text-xs shrink-0 mt-0.5">⚠</span>
                  <span className="text-[11px] text-amber-200 leading-relaxed">{r}</span>
                </li>
              ))}
            </ul>
            <div className="flex gap-3">
              <button
                onClick={() => setRevitGate(null)}
                className="flex-1 py-2 bg-slate-700 hover:bg-slate-600 text-slate-200 text-xs font-semibold rounded-lg transition-colors"
              >
                Cancel
              </button>
              <button
                onClick={_doRebuildRevit}
                className="flex-1 py-2 bg-amber-600 hover:bg-amber-500 text-white text-xs font-bold rounded-lg transition-colors"
              >
                Send Anyway
              </button>
            </div>
          </div>
        </div>
      )}

      {/* ── Project profile modal ──────────────────────────────────────────── */}
      {profileOpen && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-sm">
          <div className="bg-slate-900 border border-slate-700 rounded-2xl shadow-2xl w-[420px] p-6">
            <div className="flex items-center justify-between mb-4">
              <p className="text-white text-sm font-bold">Project Profile</p>
              <button onClick={() => setProfileOpen(false)} className="text-slate-400 hover:text-white text-sm">✕</button>
            </div>
            <p className="text-[10px] text-slate-400 mb-4 leading-relaxed">
              These values become the default dimensions when the AI cannot detect specific measurements.
              Set them once per project for better first-pass accuracy.
            </p>
            <div className="space-y-3">
              {[
                { key: 'building_type',             label: 'Building type',               isSelect: true, opts: ['commercial','residential','industrial','mixed'] },
                { key: 'typical_wall_height_mm',    label: 'Typical wall height (mm)',     isSelect: false },
                { key: 'typical_wall_thickness_mm', label: 'Typical wall thickness (mm)',  isSelect: false },
                { key: 'typical_column_size_mm',    label: 'Typical column size (mm)',     isSelect: false },
                { key: 'floor_to_floor_height_mm',  label: 'Floor-to-floor height (mm)',   isSelect: false },
                { key: 'typical_door_width_mm',     label: 'Typical door width (mm)',      isSelect: false },
                { key: 'typical_sill_height_mm',    label: 'Sill height (mm)',             isSelect: false },
              ].map(({ key, label, isSelect, opts }) => (
                <div key={key}>
                  <label className="block text-[10px] font-semibold text-slate-400 uppercase tracking-widest mb-1">{label}</label>
                  {isSelect ? (
                    <select
                      value={profileDraft[key] || ''}
                      onChange={e => setProfileDraft(p => ({ ...p, [key]: e.target.value }))}
                      className="w-full bg-slate-800 border border-slate-700 text-slate-200 text-xs rounded-lg px-3 py-2 focus:outline-none focus:ring-2 focus:ring-indigo-500"
                    >
                      {opts.map(o => <option key={o} value={o}>{o}</option>)}
                    </select>
                  ) : (
                    <input
                      type="number"
                      value={profileDraft[key] || ''}
                      onChange={e => setProfileDraft(p => ({ ...p, [key]: parseFloat(e.target.value) }))}
                      className="w-full bg-slate-800 border border-slate-700 text-slate-200 text-xs rounded-lg px-3 py-2 focus:outline-none focus:ring-2 focus:ring-indigo-500"
                    />
                  )}
                </div>
              ))}
            </div>
            <div className="flex gap-3 mt-5">
              <button onClick={() => setProfileOpen(false)}
                className="flex-1 py-2 bg-slate-700 hover:bg-slate-600 text-slate-200 text-xs font-semibold rounded-lg transition-colors">
                Cancel
              </button>
              <button onClick={handleSaveProfile} disabled={profileSaving}
                className="flex-1 py-2 bg-indigo-600 hover:bg-indigo-500 disabled:opacity-40 text-white text-xs font-bold rounded-lg transition-colors">
                {profileSaving ? 'Saving…' : 'Save Profile'}
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default Layout;
