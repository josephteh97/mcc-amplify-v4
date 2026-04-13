import React, { useState, useEffect } from 'react';

// Editable fields per element type
const FIELD_DEFS = {
  wall: [
    { key: 'thickness',    label: 'Thickness (mm)', type: 'number' },
    { key: 'height',       label: 'Height (mm)',    type: 'number' },
    { key: 'material',     label: 'Material',       type: 'text'   },
    { key: 'is_structural',label: 'Structural',     type: 'bool'   },
  ],
  column: [
    { key: 'width',    label: 'Width (mm)',  type: 'number' },
    { key: 'depth',    label: 'Depth (mm)',  type: 'number' },
    { key: 'height',   label: 'Height (mm)', type: 'number' },
    { key: 'shape',    label: 'Shape',       type: 'select', options: ['rectangular', 'circular'] },
    { key: 'material', label: 'Material',    type: 'text'   },
  ],
  door: [
    { key: 'width',     label: 'Width (mm)',  type: 'number' },
    { key: 'height',    label: 'Height (mm)', type: 'number' },
    { key: 'type_name', label: 'Type',        type: 'text'   },
  ],
  window: [
    { key: 'width',     label: 'Width (mm)',  type: 'number' },
    { key: 'height',    label: 'Height (mm)', type: 'number' },
    { key: 'type_name', label: 'Type',        type: 'text'   },
  ],
  floor: [
    { key: 'thickness', label: 'Thickness (mm)', type: 'number' },
  ],
  ceiling: [
    { key: 'thickness', label: 'Thickness (mm)', type: 'number' },
  ],
};

const TYPE_LABELS = {
  wall:    'Wall',
  column:  'Column',
  door:    'Door',
  window:  'Window',
  floor:   'Floor Slab',
  ceiling: 'Ceiling Slab',
};

const EditPanel = ({
  element,
  elementType,
  elementIndex,
  elementDefaults,
  isPatching,
  isRebuilding,
  onApply,
  onDelete,
  onClose,
  onSendToRevit,
}) => {
  const [fields, setFields] = useState({});

  // Reset local form state whenever a different element is selected
  useEffect(() => {
    if (!element) return;
    const initial = {};
    (FIELD_DEFS[elementType] || []).forEach(({ key }) => {
      initial[key] = element[key] ?? '';
    });
    setFields(initial);
  }, [element, elementType, elementIndex]);

  const handleChange = (key, value) => {
    setFields(prev => ({ ...prev, [key]: value }));
  };

  const handleApply = () => {
    const changes = {};
    (FIELD_DEFS[elementType] || []).forEach(({ key, type }) => {
      const raw = fields[key];
      if (type === 'number') {
        const n = parseFloat(raw);
        if (!isNaN(n)) changes[key] = n;
      } else if (type === 'bool') {
        changes[key] = Boolean(raw);
      } else {
        if (raw !== undefined && raw !== '') changes[key] = raw;
      }
    });
    onApply(changes);
  };

  if (!element) return null;

  const defs = FIELD_DEFS[elementType] || [];
  const typeLabel = TYPE_LABELS[elementType] || elementType;

  return (
    <div className="bg-slate-900 border border-slate-700 rounded-2xl shadow-2xl overflow-hidden">

      {/* Header */}
      <div className="flex items-center justify-between px-4 py-3 bg-slate-800 border-b border-slate-700">
        <div>
          <p className="text-[10px] font-semibold text-indigo-400 uppercase tracking-widest">
            Edit Element
          </p>
          <p className="text-sm font-bold text-white mt-0.5">
            {typeLabel} <span className="text-slate-400 font-normal text-xs">#{elementIndex}</span>
          </p>
        </div>
        <button
          onClick={onClose}
          className="w-7 h-7 flex items-center justify-center rounded-lg text-slate-400 hover:text-white hover:bg-slate-700 text-sm transition-colors"
          title="Close"
        >
          ✕
        </button>
      </div>

      {/* Current element ID chip (read-only) */}
      {element.id !== undefined && (
        <div className="px-4 pt-3">
          <span className="inline-block bg-slate-800 text-slate-400 text-[9px] font-mono px-2 py-0.5 rounded">
            id: {element.id}
          </span>
          {element.type_mark && (
            <span className="ml-1.5 inline-block bg-indigo-900/50 text-indigo-300 text-[9px] font-mono px-2 py-0.5 rounded">
              {element.type_mark}
            </span>
          )}
        </div>
      )}

      {/* Low-confidence warning */}
      {element.confidence !== undefined && element.confidence < 0.6 && (
        <div className="mx-4 mt-3 flex items-center gap-1.5 bg-amber-900/30 border border-amber-700/50 rounded-lg px-2.5 py-1.5">
          <span className="text-amber-400 text-xs shrink-0">⚠</span>
          <span className="text-[10px] text-amber-300">
            Low confidence ({(element.confidence * 100).toFixed(0)}%) — verify this detection
          </span>
        </div>
      )}

      {/* Fields */}
      <div className="px-4 py-3 space-y-3">
        {defs.map(({ key, label, type, options }) => (
          <div key={key}>
            <label className="block text-[10px] font-semibold text-slate-400 uppercase tracking-widest mb-1">
              {label}
            </label>

            {type === 'select' ? (
              <select
                value={fields[key] ?? ''}
                onChange={e => handleChange(key, e.target.value)}
                className="w-full bg-slate-800 border border-slate-700 text-slate-200 text-xs rounded-lg px-3 py-2 focus:outline-none focus:ring-2 focus:ring-indigo-500"
              >
                {(options || []).map(o => (
                  <option key={o} value={o}>{o}</option>
                ))}
              </select>
            ) : type === 'bool' ? (
              <label className="flex items-center gap-2 cursor-pointer">
                <input
                  type="checkbox"
                  checked={Boolean(fields[key])}
                  onChange={e => handleChange(key, e.target.checked)}
                  className="w-4 h-4 accent-indigo-500"
                />
                <span className="text-xs text-slate-300">Yes</span>
              </label>
            ) : (
              <input
                type={type === 'number' ? 'number' : 'text'}
                value={fields[key] ?? ''}
                onChange={e => handleChange(key, e.target.value)}
                step={type === 'number' ? '1' : undefined}
                className="w-full bg-slate-800 border border-slate-700 text-slate-200 text-xs rounded-lg px-3 py-2 focus:outline-none focus:ring-2 focus:ring-indigo-500"
              />
            )}

            {/* Firm default hint — shown when historical data differs from current value */}
            {elementDefaults?.[key] !== undefined &&
             String(elementDefaults[key]) !== String(fields[key]) && (
              <div className="flex items-center justify-between mt-1">
                <span className="text-[9px] text-indigo-400">
                  Firm default: <span className="font-mono">{String(elementDefaults[key])}</span>
                </span>
                <button
                  type="button"
                  onClick={() => handleChange(key, elementDefaults[key])}
                  className="text-[9px] text-indigo-400 underline hover:text-indigo-300 transition-colors"
                >
                  use
                </button>
              </div>
            )}
          </div>
        ))}
      </div>

      {/* Action buttons */}
      <div className="px-4 pb-4 space-y-2 border-t border-slate-800 pt-3">
        <button
          onClick={handleApply}
          disabled={isPatching || isRebuilding}
          className="w-full py-2 bg-indigo-600 hover:bg-indigo-500 disabled:opacity-40 text-white text-xs font-bold rounded-lg transition-colors"
        >
          {isPatching ? '⟳ Applying…' : 'Apply Changes'}
        </button>

        <button
          onClick={onSendToRevit}
          disabled={isPatching || isRebuilding}
          className="w-full py-2 bg-emerald-700 hover:bg-emerald-600 disabled:opacity-40 text-white text-xs font-bold rounded-lg transition-colors"
        >
          {isRebuilding ? '⟳ Sending to Revit…' : 'Send to Revit'}
        </button>

        <button
          onClick={onDelete}
          disabled={isPatching || isRebuilding}
          className="w-full py-2 bg-transparent border border-red-800 hover:bg-red-900/30 disabled:opacity-40 text-red-400 text-xs font-semibold rounded-lg transition-colors"
        >
          Delete Element
        </button>
      </div>
    </div>
  );
};

export default EditPanel;
