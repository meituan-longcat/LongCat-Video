import { useState, useEffect } from 'react'

import { 
  videometas, videometas_hs, videometas_mp, videometas_t2v_motion, videometas_t2v_surreal, videometas_t2v_style, 
  videometas_i2v, videometas_i2v_1, videometas_i2v_2,
  videometas_long_0, videometas_long_1, videometas_long_2,
  videometas_mp_0, videometas_mp_1, videometas_mp_2,
  videometas_method_0, videometas_method_1, videometas_method_2, videometas_method_3
} from './data/videometa';

import Header from './components/header';

import HeroSection from './sections/HeroSection';
import DemoSection from './sections/DemoSection';
import VideoGallerySection from './sections/VideoGallerySection';
import OverviewSection from './sections/OverviewSection';
import CapabilitiesSection from './sections/CapabilitiesSection';
import FeaturesSection from './sections/FeaturesSection';
import PlatformsSection from './sections/PlatformsSection';
import ComparisonSection from './sections/ComparisonSection';
import FAQSection from './sections/FAQSection';
import Footer from './sections/Footer';
import HSVideoGallerySection from './sections/HorizontalVideoGallerySection';
import MPVideoGallerySection from './sections/MPVideoGallerySection';
import HSRecurVideoGallerySection from './sections/HSRecurVideoGallerySection';
import HSRecurVideoGallerySection2 from './sections/HSRecurVideoGallerySection2';
import HSRecurVideoGallerySection3 from './sections/HSRecurVideoGallerySection3';
import HSRecurVideoGallerySection4 from './sections/HSRecurVideoGallerySection4';
import HSRecurVideoGallerySection5 from './sections/HSRecurVideoGallerySection5';


function App() {
  const [openFaq, setOpenFaq] = useState<number | null>(null)

  useEffect(() => {
    // Apple-style liquid glass theme with golden accents
    // document.body.style.background = 'linear-gradient(135deg, #ffffff 0%, #f8f9fa 30%, #e9ecef 70%, #ffffff 100%)'
    document.body.style.background = '#ffffff'
    document.body.style.color = '#1d1d1f'
    document.body.style.fontFamily = '-apple-system, BlinkMacSystemFont, "Segoe UI", "Roboto", "Helvetica Neue", Arial, sans-serif'
    document.body.style.margin = '0'
    document.body.style.padding = '0'
    document.body.style.minHeight = '100vh'
    document.body.style.fontWeight = '400'
    document.body.style.lineHeight = '1.47059'
    document.body.style.scrollBehavior = 'auto' // Disable default smooth scroll to use our custom one
  }, [])

  // Apple Liquid Glass design styles
  const liquidGlassCard = {
    background: 'rgba(255, 255, 255, 1)',
    border: '0.5px solid rgba(255, 255, 255, 0.2)',
    borderRadius: '12px',
    boxShadow: '0 4px 16px rgba(0, 0, 0, 0.08), 0 1px 4px rgba(0, 0, 0, 0.04)',
    backdropFilter: 'blur(12px)',
    WebkitBackdropFilter: 'blur(12px)',
    // transition: 'all 0.4s cubic-bezier(0.25, 0.46, 0.45, 0.94)',
    position: 'relative' as const
  }

  const liquidGlassCardHover = {
    ...liquidGlassCard,
    // transform: 'translateY(-4px) scale(1.02)',
    // background: 'rgba(255, 255, 255, 1)',
    // boxShadow: '0 8px 24px rgba(0, 0, 0, 0.12), 0 2px 8px rgba(0, 0, 0, 0.06)',
    // border: '0.5px solid rgba(255, 208, 0, 0.15)'
  }

  const navigationStyle = {
    position: 'sticky' as const,
    top: 0,
    zIndex: 50,
    background: 'rgba(255, 255, 255, 0.9)',
    backdropFilter: 'blur(20px)',
    WebkitBackdropFilter: 'blur(20px)',
    borderBottom: '0.5px solid rgba(255, 255, 255, 0.3)',
    padding: '16px 0',
    boxShadow: '0 1px 3px rgba(0, 0, 0, 0.08)'
  }

  const containerStyle = {
    maxWidth: '1600px',
    margin: '0 auto',
    padding: '0 24px'
  }

  const buttonStyle = {
    background: 'linear-gradient(135deg, #FFD000 0%, #F0A500 100%)',
    color: '#000000',
    padding: '16px 32px',
    borderRadius: '20px',
    border: 'none',
    fontWeight: '600',
    cursor: 'pointer',
    transition: 'all 0.4s cubic-bezier(0.25, 0.46, 0.45, 0.94)',
    fontSize: '15px',
    textTransform: 'none' as const,
    letterSpacing: '0.2px',
    boxShadow: '0 4px 16px rgba(255, 208, 0, 0.2), 0 2px 8px rgba(0, 0, 0, 0.08)',
    position: 'relative' as const
  }

  const sectionStyle = {
    padding: '100px 0',
    margin: '0'
  }

  return (
    <div style={{
      minHeight: '100vh',
      // background: 'linear-gradient(135deg, #ffffff 0%, #f8f9fa 30%, #e9ecef 70%, #ffffff 100%)',
      color: '#1d1d1f',
      position: 'relative'
    }}>


      {/* Navigation */}
      <nav style={navigationStyle}>
        <div style={containerStyle}>

          <Header 
            tabs={['Overview', 'Text-to-Video', 'Image-to-Video', 'Long Video', 'Interactive Video', 'Method']}
          />
          
        </div>
      </nav>

      {/* Hero Section */}

      <HeroSection
      />

      {/* Demo Video Section */}
      {/* <DemoSection 
        sectionStyle={sectionStyle}
        containerStyle={containerStyle}
        liquidGlassCard={liquidGlassCard}
        buttonStyle={buttonStyle}
        videoSrc="https://ext.same-assets.com/157154717/4054152294.mp4"
        videoPoster="path/to/poster.jpg"
      /> */}

      {/*Horizontal Scrollable Video Gallery */}
      {/* <HSVideoGallerySection
        sectionId={'gallery'} 
        sectionStyle={sectionStyle}
        containerStyle={containerStyle}
        liquidGlassCard={liquidGlassCard}
        liquidGlassCardHover={liquidGlassCardHover}
        videometas={videometas_hs}
      /> */}

      {/* Video Gallery */}
      <HSRecurVideoGallerySection
        sectionId={'text-to-video'} 
        sectionStyle={sectionStyle}
        containerStyle={containerStyle}
        liquidGlassCard={liquidGlassCard}
        liquidGlassCardHover={liquidGlassCardHover}
        galleryGroups={[
          { title: 'motion', videometas: videometas_t2v_motion },
          { title: 'surreal', videometas: videometas_t2v_surreal },
          { title: 'style', videometas: videometas_t2v_style },
        ]}
      />

      {/* Video Gallery */}
      <HSRecurVideoGallerySection2
        sectionId={'image-to-video'} 
        sectionStyle={sectionStyle}
        containerStyle={containerStyle}
        liquidGlassCard={liquidGlassCard}
        liquidGlassCardHover={liquidGlassCardHover}
        galleryGroups={[
          { title: '0', videometas: videometas_i2v },
          { title: '1', videometas: videometas_i2v_1 },
          { title: '2', videometas: videometas_i2v_2 },
        ]}
      />

      {/* Video Gallery */}
      <HSRecurVideoGallerySection3
        sectionId={'long video'} 
        sectionStyle={sectionStyle}
        containerStyle={containerStyle}
        liquidGlassCard={liquidGlassCard}
        liquidGlassCardHover={liquidGlassCardHover}
        galleryGroups={[
          { title: '0', videometas: videometas_long_0 },
          { title: '1', videometas: videometas_long_1 },
          { title: '2', videometas: videometas_long_2 },
        ]}
      />

      {/* Video Gallery */}
      {/* <VideoGallerySection 
        sectionId={'long video'} 
        sectionStyle={sectionStyle}
        containerStyle={containerStyle}
        liquidGlassCard={liquidGlassCard}
        liquidGlassCardHover={liquidGlassCardHover}
        videometas={videometas}
      /> */}

      {/* Video Gallery */}
      <HSRecurVideoGallerySection4
        sectionId={'interactive video'} 
        sectionStyle={sectionStyle}
        containerStyle={containerStyle}
        liquidGlassCard={liquidGlassCard}
        liquidGlassCardHover={liquidGlassCardHover}
        galleryGroups={[
          { title: '0', videometas: videometas_mp_0 },
          { title: '1', videometas: videometas_mp_1 },
          { title: '2', videometas: videometas_mp_2 },
        ]}
      />

      {/* Video Gallery */}
      <HSRecurVideoGallerySection5
        sectionId={'method'} 
        sectionStyle={sectionStyle}
        containerStyle={containerStyle}
        liquidGlassCard={liquidGlassCard}
        liquidGlassCardHover={liquidGlassCardHover}
        galleryGroups={[
          { title: 'Unified Model', videometas: videometas_method_0 },
          { title: 'Coarse-to-Fine', videometas: videometas_method_1 },
          { title: 'Multi-Reward RLHF', videometas: videometas_method_2 },
          { title: '3D Block Sparse Attention', videometas: videometas_method_3 },
        ]}
      />

      {/* Multi-Prompt Video Gallery */}
      {/* <MPVideoGallerySection 
        sectionId={'world model'} 
        sectionStyle={sectionStyle}
        containerStyle={containerStyle}
        liquidGlassCard={liquidGlassCard}
        liquidGlassCardHover={liquidGlassCardHover}
        videometas={videometas_mp}
      /> */}

      {/* Overview Section */}
      {/* <OverviewSection
        sectionId={'overview'} 
        sectionStyle={sectionStyle}
        containerStyle={containerStyle}
        liquidGlassCard={liquidGlassCard}
      /> */}

      {/* Advanced Capabilities */}
      {/* <CapabilitiesSection 
        sectionId={"capabilities"}
        sectionStyle={sectionStyle}
        containerStyle={containerStyle}
        liquidGlassCard={liquidGlassCard}
        liquidGlassCardHover={liquidGlassCardHover}
      /> */}

      {/* Complete Feature Breakdown */}
      {/* <FeaturesSection 
        sectionId={"features"}
        sectionStyle={sectionStyle}
        containerStyle={containerStyle}
        liquidGlassCard={liquidGlassCard}
      /> */}

      {/* Platform Accessibility */}
      {/* <PlatformsSection 
        sectionId={"platforms"}
        sectionStyle={sectionStyle}
        containerStyle={containerStyle}
        liquidGlassCard={liquidGlassCard}
        liquidGlassCardHover={liquidGlassCardHover}
      /> */}

      {/* Competition Analysis */}
      {/* <ComparisonSection 
        sectionId={"comparison"}
        sectionStyle={sectionStyle}
        containerStyle={containerStyle}
        liquidGlassCard={liquidGlassCard}
      /> */}

      {/* FAQ Section */}
      {/* <FAQSection 
        sectionId={"faq"}
        sectionStyle={sectionStyle}
        containerStyle={containerStyle}
        liquidGlassCard={liquidGlassCard}
      /> */}

      {/* Footer */}
      {/* <Footer 
        containerStyle={containerStyle} 
      /> */}
    </div>
  )
}

export default App
