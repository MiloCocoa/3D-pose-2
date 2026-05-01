import React from 'react';
import { 
  ShieldCheck, 
  Brain, 
  Info,
  Target
} from 'lucide-react';

const METRIC_DESCRIPTIONS: Record<string, string> = {
  "Head": "Angle between neck and vertical axis. High values indicate excessive tilt.",
  "Hip": "Lateral shift of hips from center or vertical tilt between hip joints.",
  "Frontal Knee": "Inward horizontal displacement of knees relative to ankles (Knee Valgus).",
  "Tibial Angle": "Angular deviation between the lower leg and the torso.",
  "Foot": "Vertical lift of heels or toes from the ground during the squat.",
  "Depth": "Vertical distance between hip center and knee center at the bottom.",
  "Thoracic": "AI-detected collapse or rounding of the upper spine.",
  "Trunk": "AI-detected excessive forward lean of the torso.",
  "Descent": "AI-detected mechanical instability during the downward phase.",
  "Ascent": "AI-detected mechanical instability during the upward phase."
};

export const MistakeItem = ({ label, score, threshold, type, ruleData }: any) => {
  const isMistake = score >= threshold;
  const description = METRIC_DESCRIPTIONS[label] || "No description available.";
  
  // Debug: If ruleData is missing but type is RULE, we show placeholders but alert the logic
  const hasData = ruleData && typeof ruleData.val === 'number';

  return (
    <div className={`mistake-card ${isMistake ? 'detected' : ''}`}>
      <div className="mistake-header">
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
            <span className="mistake-name">{label}</span>
            <span style={{ 
                fontSize: '0.45rem', 
                fontWeight: 900, 
                padding: '0.05rem 0.25rem', 
                borderRadius: '0.15rem',
                backgroundColor: isMistake ? 'rgba(239, 68, 68, 0.2)' : 'rgba(16, 185, 129, 0.2)',
                color: isMistake ? '#ef4444' : '#10b981',
                border: `1px solid ${isMistake ? 'rgba(239, 68, 68, 0.3)' : 'rgba(16, 185, 129, 0.3)'}`
            }}>
                {isMistake ? "DETECTED" : "PASS"}
            </span>
        </div>
        <span className="mistake-type" style={{ 
          backgroundColor: type === 'RULE' ? 'rgba(245, 158, 11, 0.1)' : 'rgba(6, 182, 212, 0.1)',
          color: type === 'RULE' ? '#f59e0b' : '#06b6d4',
          border: `1px solid ${type === 'RULE' ? 'rgba(245, 158, 11, 0.2)' : 'rgba(6, 182, 212, 0.2)'}`,
          fontSize: '0.45rem'
        }}>
          {type}
        </span>
      </div>

      <div className="progress-bar-bg">
        <div 
          className="progress-bar-fill" 
          style={{ 
            width: `${score * 100}%`,
            backgroundColor: isMistake ? '#ef4444' : '#10b981'
          }} 
        />
      </div>

      {type === 'RULE' && (
        <div className="val-limit-row">
          <span style={{ color: isMistake ? 'var(--accent-red)' : 'var(--accent-emerald)', fontWeight: 900 }}>
            VAL: {hasData ? ruleData.val.toFixed(2) : "0.00"}{ruleData?.unit || ""}
          </span>
          <span style={{ opacity: 0.5, fontWeight: 700 }}>
            LIMIT: {hasData ? ruleData.threshold.toFixed(2) : "0.00"}{ruleData?.unit || ""}
          </span>
        </div>
      )}

      <p style={{ fontSize: '0.6rem', color: 'var(--text-muted)', margin: 0, lineHeight: '1.2' }}>
          {description}
      </p>
    </div>
  );
};

export const AnalysisResults = ({ analysis }: any) => {
  const RULE_LABELS = ["Head", "Hip", "Frontal Knee", "Tibial Angle", "Foot", "Depth"];
  const AI_LABELS = ["Thoracic", "Trunk", "Descent", "Ascent"];

  if (!analysis) return null;

  return (
    <div className="right-column">
      <div className="results-top-pane">
        <div className="pane-title"><Target size={14}/> Session Summary</div>
        <div style={{ display: 'flex', gap: '2rem' }}>
          <div>
            <p style={{ fontSize: '0.6rem', color: 'var(--text-muted)', fontWeight: 900, margin: '0 0 0.15rem 0' }}>TOTAL MISTAKES</p>
            <p style={{ fontSize: '1.25rem', fontWeight: 900, margin: 0, color: (analysis.mistakes?.length || 0) > 0 ? 'var(--accent-red)' : 'var(--accent-emerald)' }}>
              {analysis.mistakes?.length || 0}
            </p>
          </div>
          <div>
            <p style={{ fontSize: '0.6rem', color: 'var(--text-muted)', fontWeight: 900, margin: '0 0 0.15rem 0' }}>OVERALL FORM</p>
            <p style={{ fontSize: '1.25rem', fontWeight: 900, margin: 0 }}>
              {!analysis.mistakes || analysis.mistakes.length === 0 ? "ELITE" : analysis.mistakes.length < 3 ? "GOOD" : "NEEDS WORK"}
            </p>
          </div>
        </div>
      </div>

      <div className="results-middle-panes">
        <div className="metadata-panel scrollable-pane" style={{ flex: 1, borderRadius: '1rem' }}>
          <div className="pane-title"><ShieldCheck size={14} color="#f59e0b"/> Safety & Geometry</div>
          {RULE_LABELS.map(label => (
            <MistakeItem 
              key={label} 
              label={label} 
              score={analysis.confidences?.[label] || 0} 
              threshold={0.5} 
              type="RULE"
              ruleData={analysis.rule_values?.[label]}
            />
          ))}
        </div>
        <div className="metadata-panel scrollable-pane" style={{ flex: 1, borderRadius: '1rem' }}>
          <div className="pane-title"><Brain size={14} color="#06b6d4"/> Postural Mechanics</div>
          {AI_LABELS.map(label => (
            <MistakeItem 
              key={label} 
              label={label} 
              score={analysis.confidences?.[label] || 0} 
              threshold={0.5} 
              type="AI"
            />
          ))}
        </div>
      </div>

      <div className="results-bottom-pane scrollable-pane">
        <div className="pane-title"><Info size={14}/> Feedback & Corrections</div>
        <ul style={{ paddingLeft: '1rem', fontSize: '0.75rem', color: 'var(--text-secondary)', lineHeight: '1.5', margin: 0 }}>
          {!analysis.mistakes || analysis.mistakes.length === 0 ? (
            <li>Form looks solid. Focus on consistent depth and tempo.</li>
          ) : (
            analysis.mistakes.map((m: string, i: number) => (
              <li key={i} style={{ marginBottom: '0.4rem' }}>
                <b style={{ color: 'var(--accent-red)', textTransform: 'uppercase' }}>{m}:</b> 
                {m === "Depth" ? " Increase hip flexion until crease is below knee top." : 
                 m === "Trunk" ? " Keep chest up and engage core to prevent excessive forward lean." :
                 " Refer to highlighted joints in 3D view for correction."}
              </li>
            ))
          )}
        </ul>
      </div>
    </div>
  );
};
