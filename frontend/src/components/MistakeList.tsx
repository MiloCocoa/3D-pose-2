import React from 'react';
import { AlertCircle, CheckCircle2, Cpu, Ruler, ShieldCheck, Activity } from 'lucide-react';

interface MistakeListProps {
  confidences: Record<string, number>;
  threshold: number;
}

const RULE_BASED_LABELS = ["Head", "Hip", "Frontal Knee", "Tibial Angle", "Foot", "Depth"];
const AI_BASED_LABELS = ["Thoracic", "Trunk", "Descent", "Ascent"];

const MistakeList: React.FC<MistakeListProps> = ({ confidences, threshold }) => {
  const renderLabelItem = (label: string, score: number) => {
    const isRule = RULE_BASED_LABELS.includes(label);
    const isMistake = score >= threshold;
    
    return (
      <div
        key={label}
        className={`group relative flex flex-col p-3 rounded-xl transition-all duration-300 ${
          isMistake 
            ? 'bg-red-500/10 border border-red-500/20 shadow-[0_0_20px_rgba(239,68,68,0.05)]' 
            : 'bg-slate-800/40 border border-slate-700/50 hover:border-emerald-500/30'
        }`}
      >
        <div className="flex items-center justify-between mb-2">
          <div className="flex items-center gap-3">
            <div className={`p-1.5 rounded-lg ${isMistake ? 'bg-red-500/20 text-red-500' : 'bg-emerald-500/20 text-emerald-500'}`}>
              {isMistake ? <AlertCircle size={14} /> : <CheckCircle2 size={14} />}
            </div>
            <span className={`font-bold text-xs tracking-tight ${isMistake ? 'text-red-100' : 'text-slate-300'}`}>
              {label.toUpperCase()}
            </span>
          </div>
          <div className={`flex items-center gap-1 text-[9px] font-black px-2 py-0.5 rounded-md border ${
            isRule ? 'text-amber-500 bg-amber-500/10 border-amber-500/20' : 'text-cyan-500 bg-cyan-500/10 border-cyan-500/20'
          }`}>
            {isRule ? <Ruler size={8}/> : <Cpu size={8}/>}
            {isRule ? "RULE" : "AI"}
          </div>
        </div>

        <div className="flex items-center justify-between gap-4">
          <div className="flex-1">
            <div className="h-1.5 w-full bg-slate-950 rounded-full overflow-hidden border border-white/5">
              <div 
                className={`h-full rounded-full transition-all duration-1000 ${isMistake ? 'bg-gradient-to-r from-red-600 to-red-400 shadow-[0_0_10px_rgba(239,68,68,0.4)]' : 'bg-gradient-to-r from-emerald-600 to-emerald-400'}`}
                style={{ width: `${score * 100}%` }}
              />
            </div>
          </div>
          <span className={`text-[10px] font-mono font-black w-12 text-right ${isMistake ? 'text-red-500' : 'text-emerald-500'}`}>
            {(score * 100).toFixed(0)}%
          </span>
        </div>
      </div>
    );
  };

  return (
    <div className="flex flex-col gap-8">
      {/* Geometric / Safety Group */}
      <section>
        <div className="flex items-center justify-between mb-4 px-1">
          <div className="flex items-center gap-2 text-amber-500">
            <ShieldCheck size={16}/>
            <h3 className="text-[10px] font-black uppercase tracking-[0.2em]">Safety & Geometry</h3>
          </div>
          <span className="text-[9px] font-bold text-slate-600 bg-slate-800 px-2 py-0.5 rounded text-white/40">DETERMINISTIC</span>
        </div>
        <div className="grid grid-cols-1 gap-2">
          {RULE_BASED_LABELS.map(label => renderLabelItem(label, confidences[label] || 0))}
        </div>
      </section>

      {/* Postural / Mechanics Group */}
      <section>
        <div className="flex items-center justify-between mb-4 px-1">
          <div className="flex items-center gap-2 text-cyan-500">
            <Activity size={16}/>
            <h3 className="text-[10px] font-black uppercase tracking-[0.2em]">Postural Mechanics</h3>
          </div>
          <span className="text-[9px] font-bold text-slate-600 bg-slate-800 px-2 py-0.5 rounded text-white/40">NEURAL NET</span>
        </div>
        <div className="grid grid-cols-1 gap-2">
          {AI_BASED_LABELS.map(label => renderLabelItem(label, confidences[label] || 0))}
        </div>
      </section>
    </div>
  );
};

export default MistakeList;
