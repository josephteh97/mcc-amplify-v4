import { useState, useEffect, useCallback, useMemo, useRef } from 'react';

/**
 * useModelSession — consolidates the job / model / human-in-the-loop /
 * project-profile state that used to live directly inside Layout.jsx as
 * 17 separate useState slots.
 *
 * Returned object:
 *   session:   read-only state (job, files, stats, recipe, …)
 *   actions:   callbacks (handleProcessingComplete, handlePatch, handleReset, …)
 *   profile:   { profile, profileDraft, profileOpen, ... } + save/open callbacks
 *   revitGate: { revitGate, requestRebuild, confirmRebuild, dismissGate }
 */
export function useModelSession() {
  // ── Core job / model ──────────────────────────────────────────────────────
  const [jobId, setJobId]       = useState(null);
  const [modelUrl, setModelUrl] = useState(null);
  const [rvtUrl, setRvtUrl]     = useState(null);
  const [fileName, setFileName] = useState('');
  const [modelReady, setModelReady] = useState(false);
  const [modelStats, setModelStats] = useState(null);
  // rvtStatus is derived from modelStats — no separate state slot.

  // ── Human-in-the-loop correction ──────────────────────────────────────────
  const [recipe, setRecipe]                   = useState(null);
  const [selectedElement, setSelectedElement] = useState(null);
  const [isPatching, setIsPatching]           = useState(false);
  const [isRebuilding, setIsRebuilding]       = useState(false);
  const [elementDefaults, setElementDefaults] = useState({});

  // ── Confidence gate ───────────────────────────────────────────────────────
  const [revitGate, setRevitGate] = useState(null);

  // ── Refs for cleanup-sensitive side effects ───────────────────────────────
  const rebuildPollRef = useRef(null);

  // ── Project profile ───────────────────────────────────────────────────────
  const [profile, setProfile]             = useState(null);
  const [profileOpen, setProfileOpen]     = useState(false);
  const [profileSaving, setProfileSaving] = useState(false);
  const [profileDraft, setProfileDraft]   = useState({});

  // ── Lifecycle ─────────────────────────────────────────────────────────────

  useEffect(() => {
    fetch('/api/project_profile')
      .then(r => r.json())
      .then(p => { setProfile(p); setProfileDraft(p); })
      .catch(() => {});
  }, []);

  useEffect(() => {
    if (!modelReady || !jobId) return;
    fetch(`/api/model/${jobId}/recipe`)
      .then(r => r.json())
      .then(setRecipe)
      .catch(err => console.warn('Recipe fetch failed:', err));
  }, [modelReady, jobId]);

  // Ensure any in-flight rebuild poll is torn down on unmount.
  useEffect(() => () => {
    if (rebuildPollRef.current) {
      clearInterval(rebuildPollRef.current);
      rebuildPollRef.current = null;
    }
  }, []);

  // ── Derived values ────────────────────────────────────────────────────────
  // Derive rvtStatus directly from the processed result; no extra state slot.
  const rvtStatus = modelStats?.rvt_status ?? null;

  // ── Derived ───────────────────────────────────────────────────────────────

  const lowConfidenceItems = useMemo(() => {
    if (!recipe) return [];
    const items = [];
    ['walls', 'columns', 'doors', 'windows', 'floors', 'ceilings'].forEach(pluralType => {
      (recipe[pluralType] || []).forEach((el, idx) => {
        if (el.confidence !== undefined && el.confidence < 0.6) {
          items.push({
            type:       pluralType.slice(0, -1),
            pluralType,
            index:      idx,
            confidence: el.confidence,
          });
        }
      });
    });
    return items.sort((a, b) => a.confidence - b.confidence);
  }, [recipe]);

  // ── Actions ───────────────────────────────────────────────────────────────

  const handleJobCreated = useCallback((id) => setJobId(id), []);

  const handleProcessingComplete = useCallback((id, result, name) => {
    setJobId(id);
    setFileName(name || 'floor_plan.pdf');
    if (result.files?.gltf) setModelUrl(`/api/download/gltf/${id}`);
    if (result.files?.rvt)  setRvtUrl(`/api/download/rvt/${id}`);
    // Fold rvt_status into modelStats so rvtStatus derives cleanly.
    const stats = {
      ...(result.stats || {}),
      rvt_status: result.rvt_status || (result.files?.rvt ? 'success' : 'skipped'),
    };
    setModelStats(stats);
    setModelReady(true);
  }, []);

  const handleReset = useCallback(() => {
    if (rebuildPollRef.current) {
      clearInterval(rebuildPollRef.current);
      rebuildPollRef.current = null;
    }
    setJobId(null);
    setModelUrl(null);
    setRvtUrl(null);
    setFileName('');
    setModelReady(false);
    setRecipe(null);
    setSelectedElement(null);
    setModelStats(null);
    setRevitGate(null);
    setIsRebuilding(false);
  }, []);

  const handleElementSelect = useCallback((type, index) => {
    if (!recipe) return;
    const pluralType = type + 's';
    const data = (recipe[pluralType] || [])[index];
    if (!data) return;
    setSelectedElement({ type, pluralType, index, data });
    fetch(`/api/corrections/defaults/${pluralType}`)
      .then(r => r.json())
      .then(setElementDefaults)
      .catch(() => setElementDefaults({}));
  }, [recipe]);

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

      // Cache-bust the glTF URL so the viewer refetches the rebuilt model.
      setModelUrl(`/api/download/gltf/${jobId}?t=${Date.now()}`);

      // PATCH now returns the updated recipe in its response body — no
      // follow-up GET needed.
      const { recipe: newRecipe } = await res.json();
      setRecipe(newRecipe);

      const updated = newRecipe[selectedElement.pluralType]?.[selectedElement.index];
      if (updated) {
        setSelectedElement(prev => ({ ...prev, data: updated }));
      } else {
        setSelectedElement(null);
      }
    } catch (err) {
      console.error('Patch failed:', err);
      alert(`Correction failed: ${err.message}`);
    } finally {
      setIsPatching(false);
    }
  }, [selectedElement, jobId]);

  const confirmRebuild = useCallback(async () => {
    if (!jobId) return;
    setRevitGate(null);
    setIsRebuilding(true);
    // Clear any prior poll (e.g. a previous rebuild attempt) before starting.
    if (rebuildPollRef.current) {
      clearInterval(rebuildPollRef.current);
      rebuildPollRef.current = null;
    }
    try {
      const res = await fetch(`/api/rebuild/${jobId}`, { method: 'POST' });
      if (!res.ok) throw new Error(await res.text());

      rebuildPollRef.current = setInterval(async () => {
        const st = await fetch(`/api/status/${jobId}`).then(r => r.json());
        if (st.status === 'completed' || st.status === 'failed') {
          clearInterval(rebuildPollRef.current);
          rebuildPollRef.current = null;
          setIsRebuilding(false);
          if (st.result?.files?.rvt) setRvtUrl(`/api/download/rvt/${jobId}`);
          if (st.result?.rvt_status) {
            setModelStats(prev => ({ ...(prev || {}), rvt_status: st.result.rvt_status }));
          }
        }
      }, 2000);
    } catch (err) {
      console.error('Rebuild failed:', err);
      setIsRebuilding(false);
      alert(`Revit rebuild failed: ${err.message}`);
    }
  }, [jobId]);

  const requestRebuild = useCallback(() => {
    if (!jobId) return;
    const blockOnFallback = profile?.gate_block_on_fallback_grid ?? true;
    const blockOnScanned  = profile?.gate_block_on_scanned_pdf   ?? true;
    const lowConfLimit    = profile?.gate_low_conf_threshold     ?? 0.30;

    const reasons = [];
    if (blockOnFallback && modelStats?.grid_confidence_label === 'Fallback')
      reasons.push('Grid detection failed — element coordinates may be unreliable.');
    if (blockOnScanned && modelStats?.is_scanned)
      reasons.push('Scanned PDF detected — vector data unavailable, accuracy reduced.');
    const totalElements = recipe
      ? Object.values(recipe).filter(Array.isArray).reduce((s, a) => s + a.length, 0)
      : 0;
    if (totalElements > 0 && lowConfidenceItems.length / totalElements > lowConfLimit)
      reasons.push(`${lowConfidenceItems.length} of ${totalElements} elements are low-confidence — consider reviewing first.`);

    if (reasons.length > 0) {
      setRevitGate({ reasons });
      return;
    }
    confirmRebuild();
  }, [jobId, modelStats, recipe, lowConfidenceItems, profile, confirmRebuild]);

  const handleSaveProfile = useCallback(async () => {
    setProfileSaving(true);
    try {
      const res = await fetch('/api/project_profile', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(profileDraft),
      });
      if (!res.ok) throw new Error(await res.text());
      setProfile(profileDraft);
      setProfileOpen(false);
    } catch (err) {
      console.error('Profile save failed:', err);
      alert(`Could not save profile: ${err.message}`);
    } finally {
      setProfileSaving(false);
    }
  }, [profileDraft]);

  return {
    // Core session
    jobId, modelUrl, rvtUrl, fileName, modelReady, modelStats, rvtStatus,

    // Human-in-the-loop
    recipe, selectedElement, isPatching, isRebuilding, elementDefaults,
    lowConfidenceItems,

    // Confidence gate
    revitGate,

    // Profile
    profile, profileOpen, profileSaving, profileDraft,

    // Setters that still need to be used directly in render
    setSelectedElement,
    setProfileOpen,
    setProfileDraft,
    setRevitGate,

    // Actions
    handleJobCreated,
    handleProcessingComplete,
    handleReset,
    handleElementSelect,
    handlePatch,
    requestRebuild,
    confirmRebuild,
    handleSaveProfile,
  };
}
