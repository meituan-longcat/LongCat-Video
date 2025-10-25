import React from 'react';

interface FeaturesSectionProps {
  sectionId: string,
  sectionStyle: React.CSSProperties;
  containerStyle: React.CSSProperties;
  liquidGlassCard: React.CSSProperties;
}

const FeaturesSection: React.FC<FeaturesSectionProps> = ({
  sectionId,
  sectionStyle,
  containerStyle,
  liquidGlassCard
}) => {

  const featureBreakdown = [
    { feature: 'Global Ranking', details: 'Ranked #2 on Artificial Analysis, above Veo 3' },
    { feature: 'Video Resolution', details: '1080P' },
    { feature: 'Max Duration', details: 'Up to 10 seconds' },
    { feature: 'Frame Rate', details: '24-30 FPS' },
    { feature: 'Physics Simulation', details: 'Ultra-realistic with accurate object interactions, fluid dynamics, natural motion patterns' },
    { feature: 'Cinematic Quality', details: 'Professional-grade visual fidelity, perfect lighting, composition' },
    { feature: 'Character Consistency', details: 'Maintains appearance across frames using advanced facial recognition and body tracking' },
    { feature: 'Motion Control', details: 'Frame-perfect precision for facial expressions and choreographed sequences' },
    { feature: 'Multilingual Support', details: 'English, Chinese, and more, with culturally appropriate visual interpretations' },
    { feature: 'Technical Specs', details: 'Powered by NCR architecture, 2.5x faster training/inference, 3x larger parameters, 4x more training data' }
  ]

  return (
    <section 
      id={sectionId}
      style={{ 
        ...sectionStyle, 
        background: 'rgba(248, 249, 250, 0.8)', 
        backdropFilter: 'blur(8px)', 
        WebkitBackdropFilter: 'blur(8px)', 
        position: 'relative', 
        zIndex: 5 
      }}
    >
      <div style={containerStyle}>
        <div style={{ textAlign: 'center', marginBottom: '60px' }}>
          <h2 style={{ 
            fontSize: '3rem', 
            fontWeight: 'bold', 
            marginBottom: '20px', 
            color: '#1d1d1f' 
          }}>
            LongCat-Video Complete Feature Breakdown
          </h2>
        </div>

        {/* Feature Breakdown Table */}
        <div style={{
          ...liquidGlassCard,
          overflow: 'hidden',
          padding: '0'
        }}>
          {/* Table Header */}
          <div style={{
            background: 'linear-gradient(135deg, #FFD000, #FFE033)',
            padding: '25px 30px',
            display: 'grid',
            gridTemplateColumns: '1fr 2fr',
            gap: '30px',
            fontWeight: 'bold',
            color: '#000000',
            fontSize: '16px'
          }}>
            <span>Feature/Capability</span>
            <span>Details</span>
          </div>

          {/* Table Rows */}
          {featureBreakdown.map((item, index) => (
            <div 
              key={item.feature}
              style={{
                padding: '20px 30px',
                display: 'grid',
                gridTemplateColumns: '1fr 2fr',
                gap: '30px',
                borderBottom: index < featureBreakdown.length - 1 
                  ? '1px solid rgba(255, 208, 0, 0.2)' 
                  : 'none',
                background: index % 2 === 0 
                  ? 'rgba(255, 208, 0, 0.05)' 
                  : 'transparent'
              }}
            >
              <span style={{ fontWeight: '700', color: '#1d1d1f' }}>
                {item.feature}
              </span>
              <span style={{ color: '#86868b' }}>
                {item.details}
              </span>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
};

export default FeaturesSection;