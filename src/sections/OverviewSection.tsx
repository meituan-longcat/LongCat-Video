import React from 'react';

interface OverviewSectionProps {
  sectionId: string;
  sectionStyle: React.CSSProperties;
  containerStyle: React.CSSProperties;
  liquidGlassCard: React.CSSProperties;
}

const OverviewSection: React.FC<OverviewSectionProps> = ({
  sectionId,
  sectionStyle,
  containerStyle,
  liquidGlassCard
}) => {
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
            LongCat-Video Overview: Revolutionary AI Video Generation
          </h2>
        </div>
        
        <div style={{ 
          display: 'grid', 
          gridTemplateColumns: '1fr 1fr', 
          gap: '60px', 
          alignItems: 'center' 
        }}>
          {/* Left Column - Description */}
          <div>
            <h3 style={{ 
              fontSize: '1.8rem', 
              fontWeight: 'bold', 
              marginBottom: '20px', 
              color: '#FFD000' 
            }}>
              What Makes LongCat-Video Special?
            </h3>
            
            <p style={{ 
              color: '#86868b', 
              lineHeight: '1.7', 
              marginBottom: '30px', 
              fontSize: '16px' 
            }}>
              LongCat-Video, developed by LongCat-Video, is a cinematic AI video generation model that has gained attention for
              its high-quality output. It's ranked #2 globally on the Artificial Analysis benchmark, surpassing
              Google's Veo 3 but trailing Seedance 1.0, making it a strong contender in the AI video space.
            </p>
            
            <div style={{ 
              display: 'flex', 
              flexDirection: 'column', 
              gap: '15px' 
            }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '15px' }}>
                <span style={{ fontSize: '28px' }}>👑</span>
                <span style={{ color: '#495057' }}>
                  <strong style={{ color: '#FFD000' }}>Global Ranking:</strong> #2 on Artificial Analysis
                </span>
              </div>
              
              <div style={{ display: 'flex', alignItems: 'center', gap: '15px' }}>
                <span style={{ fontSize: '28px' }}>🎬</span>
                <span style={{ color: '#495057' }}>
                  <strong style={{ color: '#FFD000' }}>Quality:</strong> Professional cinematic output
                </span>
              </div>
              
              <div style={{ display: 'flex', alignItems: 'center', gap: '15px' }}>
                <span style={{ fontSize: '28px' }}>⚡</span>
                <span style={{ color: '#e6e6e6' }}>
                  <strong style={{ color: '#FFD000' }}>Performance:</strong> Surpasses Google Veo 3
                </span>
              </div>
            </div>
          </div>
          
          {/* Right Column - Ranking Card */}
          <div style={{
            ...liquidGlassCard,
            padding: '40px'
          }}>
            <h4 style={{ 
              fontSize: '1.4rem', 
              fontWeight: 'bold', 
              marginBottom: '25px', 
              color: '#FFD000' 
            }}>
              Global Ranking
            </h4>
            
            <div style={{ 
              display: 'flex', 
              flexDirection: 'column', 
              gap: '20px' 
            }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <span style={{ color: '#E59500', fontWeight: '700' }}>#1 Seedance 1.0</span>
                <div style={{ 
                  width: '60px', 
                  height: '10px', 
                  background: 'linear-gradient(135deg, #E59500, #F0A500)', 
                  borderRadius: '5px', 
                  boxShadow: '0 0 10px rgba(229, 149, 0, 0.3)' 
                }} />
              </div>
              
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <span style={{ color: '#F0A500', fontWeight: '700' }}>#2 LongCat-Video 02</span>
                <div style={{ 
                  width: '50px', 
                  height: '10px', 
                  background: 'linear-gradient(135deg, #FFD000, #F0A500)', 
                  borderRadius: '5px', 
                  boxShadow: '0 0 10px rgba(255, 208, 0, 0.4)' 
                }} />
              </div>
              
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <span style={{ color: '#8b8b8b', fontWeight: '700' }}>#3 Google Veo 3</span>
                <div style={{ 
                  width: '40px', 
                  height: '10px', 
                  background: 'linear-gradient(135deg, #8b8b8b, #adb5bd)', 
                  borderRadius: '5px' 
                }} />
              </div>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
};

export default OverviewSection;