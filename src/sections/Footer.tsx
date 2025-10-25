import React from 'react';

interface FooterProps {
  containerStyle: React.CSSProperties;
}

const Footer: React.FC<FooterProps> = ({ containerStyle }) => {
  const socialLinks = [
    { name: 'Twitter', url: 'https://twitter.com/LongCat-Video' },
    { name: 'YouTube', url: 'https://youtube.com/LongCat-Video' },
    { name: 'LinkedIn', url: 'https://linkedin.com/company/LongCat-Video' }
  ];

  return (
    <footer style={{
      background: 'rgba(240, 240, 240, 0.95)',
      backdropFilter: 'blur(20px)',
      WebkitBackdropFilter: 'blur(20px)',
      borderTop: '0.5px solid rgba(255, 255, 255, 0.3)',
      padding: '48px 0',
      boxShadow: '0 -1px 3px rgba(0, 0, 0, 0.08)',
      position: 'relative',
      zIndex: 5
    }}>
      <div style={containerStyle}>
        <div style={{ textAlign: 'center' }}>
          <h3 style={{ 
            fontSize: '1.8rem', 
            fontWeight: 'bold', 
            marginBottom: '10px', 
            color: '#1d1d1f' 
          }}>
            LongCat-Video - Next-Generation AI Video Creation
          </h3>
          
          <p style={{ 
            color: '#86868b', 
            marginBottom: '30px', 
            fontSize: '16px' 
          }}>
            Experience the power of #2 globally ranked AI video generation model
          </p>
          
          <div style={{ 
            display: 'flex', 
            justifyContent: 'center', 
            gap: '20px', 
            marginBottom: '30px',
          }}>
            {socialLinks.map((social) => (
              <a
                key={social.name}
                href={social.url}
                style={{
                  color: '#FFD000',
                  textDecoration: 'none',
                  padding: '12px 24px',
                  background: 'rgba(255, 208, 0, 0.1)',
                  border: '1px solid #FFD000',
                  borderRadius: '15px',
                  transition: 'all 0.3s ease',
                  fontWeight: '600',
                  minWidth: '120px'
                }}
                onMouseEnter={(e) => {
                  e.currentTarget.style.background = '#FFD000';
                  e.currentTarget.style.color = '#000000';
                }}
                onMouseLeave={(e) => {
                  e.currentTarget.style.background = 'rgba(255, 208, 0, 0.1)';
                  e.currentTarget.style.color = '#FFD000';
                }}
                target="_blank"
                rel="noopener noreferrer nofollow"
              >
                {social.name}
              </a>
            ))}
          </div>
          
          <p style={{ 
            fontSize: '14px', 
            color: '#adb5bd',
            marginTop: '20px'
          }}>
            © {new Date().getFullYear()} LongCat-Video Information Hub. All external links open in new tabs with nofollow attributes.
          </p>
        </div>
      </div>
    </footer>
  );
};

export default Footer;