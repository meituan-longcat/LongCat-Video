import React, { useState } from 'react';

interface FAQSectionProps {
  sectionId: string;
  sectionStyle: React.CSSProperties;
  containerStyle: React.CSSProperties;
  liquidGlassCard: React.CSSProperties;
}

const FAQSection: React.FC<FAQSectionProps> = ({
  sectionId,
  sectionStyle,
  containerStyle,
  liquidGlassCard,
}) => {
  const [openFaq, setOpenFaq] = useState<number | null>(null);

  const faqData = [
    {
      question: 'What is LongCat-Video and who developed it?',
      answer: 'LongCat-Video is a cinematic AI video generation model developed by LongCat-Video. It\'s currently ranked #2 globally on the Artificial Analysis benchmark, surpassing Google\'s Veo 3. The model specializes in creating professional-grade videos with ultra-realistic physics simulation and cinematic quality output.'
    },
    {
      question: 'How much does it cost to generate videos with LongCat-Video?',
      answer: 'LongCat-Video costs $0.28 per video generation through fal\'s API platform. On BasedLabs, it costs 300 credits per video. This makes it highly cost-efficient compared to other professional AI video generation models while maintaining world-class quality.'
    },
    {
      question: 'What are the technical specifications of LongCat-Video?',
      answer: 'LongCat-Video generates videos in 1080P resolution with up to 10 seconds duration at 24-30 FPS. It\'s powered by NCR architecture with 2.5x faster training and inference, 3x larger parameters, and 4x more training data compared to previous versions. The model supports both text-to-video and image-to-video generation.'
    },
    {
      question: 'Where can I access and use LongCat-Video?',
      answer: 'You can access LongCat-Video through multiple platforms: the official LongCat-Video AI website (LongCat-Video.video), BasedLabs platform which supports 70+ languages, and fal\'s API platform for developers. Each platform offers different features and pricing options to suit various user needs.'
    },
    {
      question: 'How does LongCat-Video compare to Google Veo 3 and other competitors?',
      answer: 'LongCat-Video ranks #2 globally, surpassing Google Veo 3 (#3) but trailing Seedance 1.0 (#1). It offers superior physics simulation, more flexible content policies, competitive pricing, and wider platform integration compared to many competitors. The model excels in cinematic quality and character consistency.'
    },
    {
      question: 'What makes LongCat-Video\'s physics simulation special?',
      answer: 'LongCat-Video features ultra-realistic physics simulation that accurately handles object interactions, fluid dynamics, and natural motion patterns. It can even handle extreme physics scenarios like acrobatics while maintaining realistic movement and collision detection, setting it apart from many other AI video models.'
    },
    {
      question: 'What languages does LongCat-Video support?',
      answer: 'LongCat-Video supports multiple languages including English and Chinese, with culturally appropriate visual interpretations. Through BasedLabs platform, it supports over 70 languages, making it accessible to a global user base for international content creation.'
    },
    {
      question: 'How does character consistency work in LongCat-Video?',
      answer: 'LongCat-Video uses advanced facial recognition and body tracking technology to maintain consistent character appearances across frames. This addresses a common challenge in AI video generation, ensuring that characters look the same throughout the video with frame-perfect motion precision for facial expressions and choreographed sequences.'
    },
    {
      question: 'What types of content can I create with LongCat-Video?',
      answer: 'LongCat-Video supports diverse content creation including action sequences, animation, fantasy worlds, natural landscapes, character portraits, cinematic storytelling, creative narratives, and physics demonstrations. It\'s suitable for film production, marketing content, educational videos, and social media creation with professional-grade quality.'
    },
    {
      question: 'How do I get started with LongCat-Video on BasedLabs?',
      answer: 'Getting started is simple: 1) Click "Generate" on the BasedLabs tool page to launch the editor, 2) Select "Text to Video" and choose "LongCat-Video LongCat-Video AI" as your model, 3) Enter your text prompt or upload an image, then click "Generate". The process costs 300 credits per video and supports both text-to-video and image-to-video generation.'
    }
  ]

  return (
    <section id={sectionId} style={{ ...sectionStyle, position: 'relative', zIndex: 5 }}>
      <div style={containerStyle}>
        <div style={{ textAlign: 'center', marginBottom: '60px' }}>
          <h2 style={{ 
            fontSize: '3rem', 
            fontWeight: 'bold', 
            marginBottom: '20px', 
            color: '#1d1d1f' 
          }}>
            Frequently Asked Questions About LongCat-Video
          </h2>
        </div>

        <div style={{ maxWidth: '900px', margin: '0 auto' }}>
          {faqData.map((faq, index) => (
            <div 
              key={faq.question} 
              style={{
                ...liquidGlassCard,
                marginBottom: '20px',
                overflow: 'hidden',
                padding: '0'
              }}
            >
              <button
                style={{
                  width: '100%',
                  padding: '30px',
                  textAlign: 'left',
                  display: 'flex',
                  justifyContent: 'space-between',
                  alignItems: 'center',
                  backgroundColor: 'transparent',
                  border: 'none',
                  color: '#1d1d1f',
                  cursor: 'pointer',
                  fontSize: '18px',
                  fontWeight: '700',
                  transition: 'all 0.3s ease'
                }}
                onClick={() => setOpenFaq(openFaq === index ? null : index)}
                onMouseEnter={(e) => {
                  e.currentTarget.style.background = 'rgba(255, 208, 0, 0.15)';
                }}
                onMouseLeave={(e) => {
                  e.currentTarget.style.background = 'transparent';
                }}
                aria-expanded={openFaq === index}
                aria-controls={`faq-content-${index}`}
              >
                <span>{faq.question}</span>
                <span style={{
                  transform: openFaq === index ? 'rotate(180deg)' : 'rotate(0deg)',
                  transition: 'transform 0.3s ease',
                  fontSize: '24px',
                  color: '#FFD000'
                }}>
                  ▼
                </span>
              </button>

              {openFaq === index && (
                <div 
                  id={`faq-content-${index}`}
                  style={{
                    padding: '0 30px 30px',
                    color: '#86868b',
                    lineHeight: '1.7',
                    background: 'rgba(255, 208, 0, 0.1)',
                    borderTop: '1px solid rgba(255, 208, 0, 0.2)',
                    fontSize: '15px'
                  }}
                >
                  {faq.answer}
                </div>
              )}
            </div>
          ))}
        </div>
      </div>
    </section>
  );
};

export default FAQSection;