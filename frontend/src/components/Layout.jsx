import React from 'react';
import UploadPanel from './UploadPanel';
import ChatPanel from './ChatPanel';
import Viewer from './Viewer';
import EditPanel from './EditPanel';
import { useModelSession } from '../hooks/useModelSession';

const Layout = () => {
  const s = useModelSession();

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

          <div className="px-4 pt-4 pb-2">
            <p className="text-[11px] font-semibold text-slate-400 uppercase tracking-widest">Upload</p>
          </div>

          <div className="px-4 pb-4">
            <UploadPanel
              onJobCreated={s.handleJobCreated}
              onProcessingComplete={s.handleProcessingComplete}
            />
          </div>

          {/* Model stats — shown when model is ready */}
          {s.modelReady && s.modelStats && (
            <div className="px-4 pb-4 space-y-2">
              <p className="text-[11px] font-semibold text-slate-400 uppercase tracking-widest mb-2">Analysis</p>

              {/* Scanned PDF warning */}
              {s.modelStats.is_scanned && (
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
                  s.modelStats.grid_confidence_label === 'High'     ? 'bg-emerald-900/60 text-emerald-300' :
                  s.modelStats.grid_confidence_label === 'Medium'   ? 'bg-yellow-900/60 text-yellow-300'  :
                  s.modelStats.grid_confidence_label === 'Fallback' ? 'bg-red-900/60 text-red-300'        :
                                                                      'bg-slate-700 text-slate-400'
                }`}>
                  {s.modelStats.grid_confidence_label} ({(s.modelStats.grid_confidence * 100).toFixed(0)}%)
                </span>
              </div>

              {/* RVT export status badge */}
              {s.rvtStatus && (
                <div className="flex items-center justify-between bg-slate-800 rounded-lg px-3 py-2">
                  <span className="text-[10px] text-slate-400">Revit export</span>
                  <span className={`text-[10px] font-bold px-2 py-0.5 rounded-full ${
                    s.rvtStatus === 'success'            ? 'bg-emerald-900/60 text-emerald-300' :
                    s.rvtStatus === 'warnings_accepted'  ? 'bg-yellow-900/60 text-yellow-300'   :
                    s.rvtStatus === 'skipped'            ? 'bg-slate-700 text-slate-400'        :
                                                           'bg-red-900/60 text-red-300'
                  }`}>
                    {s.rvtStatus === 'success'           ? 'Success'          :
                     s.rvtStatus === 'warnings_accepted' ? 'Warnings kept'    :
                     s.rvtStatus === 'skipped'           ? 'Skipped'          :
                                                           'Failed'}
                  </span>
                </div>
              )}

              {/* Element counts */}
              <div className="bg-slate-800 rounded-lg px-3 py-2 space-y-1">
                <div className="flex justify-between">
                  <span className="text-[10px] text-slate-400">Elements detected</span>
                  <span className="text-[10px] font-semibold text-slate-200">{s.modelStats.element_count}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-[10px] text-slate-400">Grid lines</span>
                  <span className="text-[10px] font-semibold text-slate-200">{s.modelStats.grid_lines}</span>
                </div>
              </div>

              {/* Pre-clash validation warnings */}
              {s.modelStats.validation_warnings?.length > 0 && (
                <div className="bg-red-900/20 border border-red-800/50 rounded-lg px-3 py-2 space-y-1.5">
                  <p className="text-[10px] font-semibold text-red-400 uppercase tracking-widest">
                    {s.modelStats.validation_warnings.length} Clash Warning{s.modelStats.validation_warnings.length > 1 ? 's' : ''}
                  </p>
                  {s.modelStats.validation_warnings.map((w, i) => (
                    <p key={i} className="text-[9px] text-red-300 leading-relaxed">• {w}</p>
                  ))}
                </div>
              )}
            </div>
          )}

          {/* Low-confidence elements — clickable to select for review */}
          {s.modelReady && s.lowConfidenceItems.length > 0 && (
            <div className="px-4 pb-4 space-y-2">
              <p className="text-[11px] font-semibold text-amber-500 uppercase tracking-widest mb-2">
                ⚠ Needs Review ({s.lowConfidenceItems.length})
              </p>
              <div className="space-y-1">
                {s.lowConfidenceItems.map((item, i) => (
                  <button
                    key={i}
                    onClick={() => s.handleElementSelect(item.type, item.index)}
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
          {s.modelReady && (
            <div className="px-4 pb-4 space-y-2">
              <p className="text-[11px] font-semibold text-slate-400 uppercase tracking-widest mb-3">Downloads</p>

              {s.rvtUrl && (
                <a
                  href={s.rvtUrl}
                  download
                  className="flex items-center gap-2 w-full bg-indigo-600 hover:bg-indigo-500 text-white text-xs font-semibold px-4 py-2.5 rounded-lg transition-colors"
                >
                  <span>⬇</span> Download RVT
                </a>
              )}
              {s.modelUrl && (
                <a
                  href={s.modelUrl}
                  download
                  className="flex items-center gap-2 w-full bg-blue-600 hover:bg-blue-500 text-white text-xs font-semibold px-4 py-2.5 rounded-lg transition-colors"
                >
                  <span>⬇</span> Download glTF
                </a>
              )}
              <button
                onClick={s.handleReset}
                className="flex items-center gap-2 w-full bg-slate-700 hover:bg-slate-600 text-slate-200 text-xs font-semibold px-4 py-2.5 rounded-lg transition-colors"
              >
                ↺ New Upload
              </button>
            </div>
          )}

          {/* Project settings button */}
          <div className="px-4 pb-3">
            <button
              onClick={() => s.setProfileOpen(true)}
              className="w-full flex items-center gap-2 bg-slate-800 hover:bg-slate-700 border border-slate-700 text-slate-300 text-xs font-medium px-3 py-2 rounded-lg transition-colors"
            >
              <span>⚙</span>
              <span>Project Profile</span>
              {s.profile?.building_type && (
                <span className="ml-auto text-[9px] text-slate-500 capitalize">{s.profile.building_type}</span>
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
          {s.modelReady && (
            <div className="absolute top-3 left-3 z-10">
              <div className="flex items-center gap-2 bg-slate-900/80 backdrop-blur-sm border border-slate-700 rounded-xl px-3 py-1.5">
                <span className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse"></span>
                <span className="text-xs font-semibold text-slate-200 max-w-[180px] truncate">{s.fileName}</span>
              </div>
            </div>
          )}

          {/* Click hint — shown when model is ready and nothing is selected */}
          {s.modelReady && !s.selectedElement && (
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
            modelUrl={s.modelUrl}
            onElementSelect={s.handleElementSelect}
            selectedMesh={s.selectedElement}
          />

          {/* EditPanel — absolute overlay, top-right of center column */}
          {s.selectedElement && (
            <div className="absolute top-3 right-3 z-20 w-72">
              <EditPanel
                element={s.selectedElement.data}
                elementType={s.selectedElement.type}
                elementIndex={s.selectedElement.index}
                elementDefaults={s.elementDefaults}
                isPatching={s.isPatching}
                isRebuilding={s.isRebuilding}
                onApply={s.handlePatch}
                onDelete={() => s.handlePatch({}, true)}
                onClose={() => s.setSelectedElement(null)}
                onSendToRevit={s.requestRebuild}
              />
            </div>
          )}
        </main>

        {/* ── RIGHT: Chat agent ───────────────────────────────────── */}
        <aside className="w-80 shrink-0 flex flex-col bg-slate-900 border-l border-slate-800">
          <div className="px-4 pt-4 pb-2 shrink-0">
            <p className="text-[11px] font-semibold text-slate-400 uppercase tracking-widest">AI Supervisor</p>
          </div>
          <div className="flex-1 min-h-0 px-3 pb-3">
            <div className="h-full">
              <ChatPanel jobId={s.jobId} />
            </div>
          </div>
        </aside>

      </div>

      {/* ── Confidence gate modal ──────────────────────────────────────────── */}
      {s.revitGate && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-sm">
          <div className="bg-slate-900 border border-amber-700/60 rounded-2xl shadow-2xl w-[400px] p-6">
            <p className="text-amber-400 text-sm font-bold mb-1">Quality Warning</p>
            <p className="text-slate-300 text-xs mb-4 leading-relaxed">
              The model may have reliability issues. Review before committing to Revit:
            </p>
            <ul className="space-y-2 mb-5">
              {s.revitGate.reasons.map((r, i) => (
                <li key={i} className="flex items-start gap-2 bg-amber-900/20 border border-amber-800/40 rounded-lg px-3 py-2">
                  <span className="text-amber-400 text-xs shrink-0 mt-0.5">⚠</span>
                  <span className="text-[11px] text-amber-200 leading-relaxed">{r}</span>
                </li>
              ))}
            </ul>
            <div className="flex gap-3">
              <button
                onClick={() => s.setRevitGate(null)}
                className="flex-1 py-2 bg-slate-700 hover:bg-slate-600 text-slate-200 text-xs font-semibold rounded-lg transition-colors"
              >
                Cancel
              </button>
              <button
                onClick={s.confirmRebuild}
                className="flex-1 py-2 bg-amber-600 hover:bg-amber-500 text-white text-xs font-bold rounded-lg transition-colors"
              >
                Send Anyway
              </button>
            </div>
          </div>
        </div>
      )}

      {/* ── Project profile modal ──────────────────────────────────────────── */}
      {s.profileOpen && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-sm">
          <div className="bg-slate-900 border border-slate-700 rounded-2xl shadow-2xl w-[420px] p-6">
            <div className="flex items-center justify-between mb-4">
              <p className="text-white text-sm font-bold">Project Profile</p>
              <button onClick={() => s.setProfileOpen(false)} className="text-slate-400 hover:text-white text-sm">✕</button>
            </div>
            <p className="text-[10px] text-slate-400 mb-4 leading-relaxed">
              These values become the default dimensions when the AI cannot detect specific measurements.
              Set them once per project for better first-pass accuracy.
            </p>
            <div className="space-y-3 max-h-[60vh] overflow-y-auto pr-1">
              {[
                { key: 'building_type',             label: 'Building type',               kind: 'select', opts: ['commercial','residential','industrial','mixed'] },
                { key: 'typical_wall_height_mm',    label: 'Typical wall height (mm)',     kind: 'number' },
                { key: 'typical_wall_thickness_mm', label: 'Typical wall thickness (mm)',  kind: 'number' },
                { key: 'typical_column_size_mm',    label: 'Typical column size (mm)',     kind: 'number' },
                { key: 'floor_to_floor_height_mm',  label: 'Floor-to-floor height (mm)',   kind: 'number' },
                { key: 'typical_door_width_mm',     label: 'Typical door width (mm)',      kind: 'number' },
                { key: 'typical_sill_height_mm',    label: 'Sill height (mm)',             kind: 'number' },
                { key: '__divider',                 label: 'Revit Commit Gate',            kind: 'divider' },
                { key: 'gate_block_on_fallback_grid', label: 'Warn on fallback grid',      kind: 'toggle' },
                { key: 'gate_block_on_scanned_pdf',   label: 'Warn on scanned PDF',        kind: 'toggle' },
                { key: 'gate_low_conf_threshold',     label: 'Low-confidence threshold (0–1)', kind: 'number', step: 0.05 },
              ].map(({ key, label, kind, opts, step }) => (
                kind === 'divider' ? (
                  <p key={key} className="text-[10px] font-bold text-slate-500 uppercase tracking-widest pt-2 border-t border-slate-800">{label}</p>
                ) : (
                  <div key={key}>
                    <label className="block text-[10px] font-semibold text-slate-400 uppercase tracking-widest mb-1">{label}</label>
                    {kind === 'select' ? (
                      <select
                        value={s.profileDraft[key] || ''}
                        onChange={e => s.setProfileDraft(p => ({ ...p, [key]: e.target.value }))}
                        className="w-full bg-slate-800 border border-slate-700 text-slate-200 text-xs rounded-lg px-3 py-2 focus:outline-none focus:ring-2 focus:ring-indigo-500"
                      >
                        {opts.map(o => <option key={o} value={o}>{o}</option>)}
                      </select>
                    ) : kind === 'toggle' ? (
                      <label className="flex items-center gap-2 text-xs text-slate-300 cursor-pointer">
                        <input
                          type="checkbox"
                          checked={!!s.profileDraft[key]}
                          onChange={e => s.setProfileDraft(p => ({ ...p, [key]: e.target.checked }))}
                          className="w-4 h-4 accent-indigo-500"
                        />
                        <span>{s.profileDraft[key] ? 'Enabled' : 'Disabled'}</span>
                      </label>
                    ) : (
                      <input
                        type="number"
                        step={step ?? 1}
                        value={s.profileDraft[key] ?? ''}
                        onChange={e => s.setProfileDraft(p => ({ ...p, [key]: parseFloat(e.target.value) }))}
                        className="w-full bg-slate-800 border border-slate-700 text-slate-200 text-xs rounded-lg px-3 py-2 focus:outline-none focus:ring-2 focus:ring-indigo-500"
                      />
                    )}
                  </div>
                )
              ))}
            </div>
            <div className="flex gap-3 mt-5">
              <button onClick={() => s.setProfileOpen(false)}
                className="flex-1 py-2 bg-slate-700 hover:bg-slate-600 text-slate-200 text-xs font-semibold rounded-lg transition-colors">
                Cancel
              </button>
              <button onClick={s.handleSaveProfile} disabled={s.profileSaving}
                className="flex-1 py-2 bg-indigo-600 hover:bg-indigo-500 disabled:opacity-40 text-white text-xs font-bold rounded-lg transition-colors">
                {s.profileSaving ? 'Saving…' : 'Save Profile'}
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default Layout;
