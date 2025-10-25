import React from 'react';

interface ComparisonSectionProps {
  sectionId: string;
  sectionStyle: React.CSSProperties;
  containerStyle: React.CSSProperties;
  liquidGlassCard: React.CSSProperties;
}

const ComparisonSection: React.FC<ComparisonSectionProps> = ({
  sectionId,
  sectionStyle,
  containerStyle,
  liquidGlassCard,
}) => {

  const competitiveAdvantages = [
    'Surpasses Google Veo 3 in quality metrics',
    'More flexible content policies than competitors',
    'Superior physics simulation capabilities',
    'Competitive pricing at $0.28 per video',
    'Wide platform integration availability',
    'Professional-grade cinematic output'
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
            LongCat-Video vs Industry Competition
          </h2>
        </div>

        {/* Competitive Advantages Card */}
        <div style={{
          ...liquidGlassCard,
          padding: '40px',
          marginBottom: '60px'
        }}>
          <h3 style={{ 
            fontSize: '1.8rem', 
            fontWeight: 'bold', 
            marginBottom: '30px', 
            color: '#FFD000',
          }}>
            Why Choose LongCat-Video?
          </h3>
          
          <div style={{ 
            display: 'grid', 
            gridTemplateColumns: 'repeat(auto-fit, minmax(350px, 1fr))', 
            gap: '20px' 
          }}>
            {competitiveAdvantages.map((advantage) => (
              <div 
                key={advantage} 
                style={{ 
                  display: 'flex', 
                  alignItems: 'center', 
                  gap: '15px',
                }}
              >
                <span style={{ 
                  color: '#FFD000', 
                  fontSize: '22px', 
                  fontWeight: 'bold',
                }}>
                  ✓
                </span>
                <span style={{ 
                  color: '#495057', 
                }}>
                  {advantage}
                </span>
              </div>
            ))}
          </div>
        </div>

        {/* Comparison Table (Optional - can be added later) */}
        {/* <ComparisonTable /> */}
      </div>
    </section>
  );
};

export default ComparisonSection;