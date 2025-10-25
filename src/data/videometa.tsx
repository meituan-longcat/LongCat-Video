export interface VideoMeta {
    // id: string;
    label: string;
    title: string;
    url: string;
    prompt?: string;
  }
  
const videometas: VideoMeta[] = [
    {
        label: 'Action', 
        title: 'Dynamic Action Sequences', 
        url: 'https://s3plus-bj02.sankuai.com/v3d-personal/liangshijun/demo-website/long/0008-%E8%A7%86%E9%A2%91%E9%87%87%E7%94%A8%E8%88%AA%E6%8B%8D%E8%A7%86%E8%A7%92%E5%B1%95%E7%8E%B0%E8%87%AA%E7%84%B6%E5%B3%A1%E8%B0%B7%E6%99%AF%E8%A7%82%EF%BC%8C%E9%95%9C%E5%A4%B4%E5%BF%AB%E9%80%9F%E7%A9%BF%E6%A2%AD%E4%BA%8E%E4%B8%A4%E4%BE%A7%E8%A6%86%E7%9B%96%E8%8B%94%E8%97%93%E7%9A%84%E9%99%A1%E5%B3%AD%E5%B2%A9%E5%A3%81%E4%B9%8B%E9%97%B4%EF%BC%8C%E8%BF%87%E7%A8%8B%E4%B8%AD%E4%BC%B4%E9%9A%8F%E6%97%8B%E8%BD%AC%E3%80%81%E4%BF%AF%E5%86%B2%E7%AD%89%E8%BF%90.mp4', 
        prompt: '视频采用航拍视角展现自然峡谷景观，镜头快速穿梭于两侧覆盖苔藓的陡峭岩壁之间，过程中伴随旋转、俯冲等运镜动作，清晰呈现峡谷底部流淌的河流、岩壁上的纹理，以及山间一条橙红色小径；画面以自然光呈现，天空多云，整体为真实拍摄的户外自然场景，通过动态的镜头运动展现峡谷的险峻与地貌细节。' 
    },
    {
        label: 'Action', 
        title: 'Dynamic Action Sequences', 
        url: 'https://s3plus-bj02.sankuai.com/v3d-personal/liangshijun/demo-website/long/0013-%E4%B8%80%E4%B8%AA%E8%BA%AB%E7%9D%80%E6%B5%85%E7%81%B0%E8%89%B2%E5%A4%96%E5%A5%97%E3%80%81%E6%B7%B1%E8%89%B2%E8%A3%A4%E5%AD%90%E5%B9%B6%E4%BD%A9%E6%88%B4%E6%89%8B%E5%A5%97%E7%9A%84%E4%BA%BA%EF%BC%8C%E5%9C%A8%E4%B8%A4%E4%BE%A7%E6%9C%89%E7%A7%AF%E9%9B%AA%E8%A6%86%E7%9B%96%E5%B1%B1%E4%BD%93%E4%B8%8E%E6%8A%A4%E6%A0%8F%E7%9A%84%E5%BC%80%E9%98%94%E5%85%AC%E8%B7%AF%E4%B8%8A%E9%AB%98%E9%80%9F%E6%BB%91%E8%A1%8C%E9%95%BF%E6%9D%BF%EF%BC%9B%E6%AD%A4%E4%BA%BA.mp4', 
        prompt: '一个身着浅灰色外套、深色裤子并佩戴手套的人，在两侧有积雪覆盖山体与护栏的开阔公路上高速滑行长板；此人全程保持弯腰俯身的姿势，借助长板快速向前；背景中，左侧岩壁与积雪交替闪过，右侧护栏沿公路延伸，远处连绵雪山和飘着粉白云彩的天空构成开阔自然景观；镜头以中景跟拍视角，平稳跟随滑板者的滑行轨迹，整体画面为实际外景拍摄，自然光线柔和地笼罩场景，呈现出运动过程中与壮美自然风光融合的视觉效果。' 
    },
    {
        label: 'Action', 
        title: 'Dynamic Action Sequences', 
        url: 'https://s3plus-bj02.sankuai.com/v3d-personal/liangshijun/demo-website/long/0000-%E5%A4%9C%E6%99%9A%E7%9A%84%E5%9F%8E%E5%B8%82%E9%81%93%E8%B7%AF%E4%B8%8A%EF%BC%8C%E5%A4%9A%E8%BE%86%E6%B1%BD%E8%BD%A6%E5%9C%A8%E8%BD%A6%E9%81%93%E8%A1%8C%E9%A9%B6%EF%BC%8C%E5%8F%B3%E4%BE%A7%E6%9C%89%E4%B8%80%E8%BE%86%E6%A9%99%E8%89%B2%E5%87%BA%E7%A7%9F%E8%BD%A6%EF%BC%9B%E9%95%9C%E5%A4%B4%E4%BB%A5%E8%B7%9F%E9%9A%8F%E8%A1%8C%E9%A9%B6%E7%9A%84%E8%A7%86%E8%A7%92%E5%90%91%E5%89%8D%E6%8E%A8%E8%BF%9B%EF%BC%8C%E5%91%88%E7%8E%B0%E5%87%BA%E9%81%93%E8%B7%AF.mp4', 
        prompt: '夜晚的城市道路上，多辆汽车在车道行驶，右侧有一辆橙色出租车；镜头以跟随行驶的视角向前推进，呈现出道路两旁林立的高楼、亮着光的路灯与建筑、绿化带里的树木和花丛，空中还悬着一轮明月，车辆在标线清晰的路面有序前行，灯光以夜景下的路灯和建筑灯光为主。' 
    },
    {
        label: 'Action', 
        title: 'Dynamic Action Sequences', 
        url: 'https://s3plus-bj02.sankuai.com/v3d-personal/liangshijun/demo-website/long/0022-%E4%B8%80%E4%BD%8D%E9%95%BF%E5%8F%91%E7%9A%84%E5%B9%B4%E8%BD%BB%E5%A5%B3%E6%80%A7%E7%A9%BF%E7%9D%80%E7%99%BD%E8%89%B2%E6%97%A0%E8%A2%96%E4%B8%8A%E8%A1%A3%EF%BC%8C%E5%9D%90%E5%9C%A8%E8%A3%85%E9%A5%B0%E6%9C%89%E6%B5%85%E8%89%B2%E6%8A%A4%E5%A2%99%E6%9D%BF%E7%9A%84%E6%88%BF%E9%97%B4%E9%87%8C%EF%BC%8C%E5%A5%B9%E6%89%8B%E6%8C%81%E4%B8%80%E7%93%B6%E9%A6%99%E6%B0%B4%EF%BC%8C%E8%BE%B9%E8%AF%B4%E8%AF%9D%E8%BE%B9%E8%BD%AC%E5%8A%A8%E9%A6%99%E6%B0%B4%E7%93%B6.mp4', 
        prompt: '一位长发的年轻女性穿着白色无袖上衣，坐在装饰有浅色护墙板的房间里，她手持一瓶香水，边说话边转动香水瓶向镜头展示瓶身；画面右侧有一盏亮着暖光的台灯和叠放的书籍，左侧可见毛绒靠垫，整体采用中景固定镜头、平拍角度拍摄，灯光为暖调室内光，呈现实际拍摄的真人画面风格。' 
    },
    {
        label: 'Action', 
        title: 'Dynamic Action Sequences', 
        url: 'https://s3plus-bj02.sankuai.com/v3d-personal/liangshijun/demo-website/long/0015-realistic%20photography%2C%20A%20mukbang%20influencer%20sittin.mp4', 
        prompt: 'realistic photography, A mukbang influencer sitting at a table piled high with a variety of sushi, sashimi, and tempura, ready to indulge in the spread. They begin by dipping a piece of sushi into soy sauce, slowly bringing it to their mouth and taking a big bite. The satisfying crunch of the tempura contrasts with the soft, delicate fish of the sushi, and each bite is followed by a delighted expression. As they eat, they interact with their followers, responding to comments and laughing at jokes, making the experience feel like a cozy, virtual dinner party. The gentle clink of chopsticks and the sound of eating create an oddly soothing atmosphere.' 
    },
    {
        label: 'Action', 
        title: 'Dynamic Action Sequences', 
        url: 'https://s3plus-bj02.sankuai.com/v3d-personal/liangshijun/demo-website/long/0008-realistic%20photography%2C%20In%20a%20sunlit%20dance%20studio%20wi.mp4', 
        prompt: 'realistic photography, In a sunlit dance studio with high ceilings and expansive mirrors, a graceful ballerina performs a solo routine. Her lithe body moves with fluid precision as she executes a perfect pirouette, her tutu flaring out around her. Her arms arc elegantly above her head, fingers extended in delicate curves. The dancer\'s face is a picture of serene concentration, her eyes focused on a distant point. Sunlight streams through tall windows, casting long shadows across the polished wooden floor and highlighting the dust motes that swirl in her wake. The room is silent save for the soft whisper of her pointe shoes as they brush against the floor, emphasizing the ethereal quality of her movements.' 
    },

];

const videometas_t2v_motion: VideoMeta[] = [
    {
        label: 'Action', 
        title: 'Dynamic Action Sequences', 
        url: 'assets/videos/t2v/motion-01.mp4', 
        // prompt: '视频采用航拍视角展现自然峡谷景观，镜头快速穿梭于两侧覆盖苔藓的陡峭岩壁之间，过程中伴随旋转、俯冲等运镜动作，清晰呈现峡谷底部流淌的河流、岩壁上的纹理，以及山间一条橙红色小径；画面以自然光呈现，天空多云，整体为真实拍摄的户外自然场景，通过动态的镜头运动展现峡谷的险峻与地貌细节。' 
    },
    {
        label: 'Action', 
        title: 'Dynamic Action Sequences', 
        url: 'assets/videos/t2v/motion-02.mp4', 
        // prompt: '一个身着浅灰色外套、深色裤子并佩戴手套的人，在两侧有积雪覆盖山体与护栏的开阔公路上高速滑行长板；此人全程保持弯腰俯身的姿势，借助长板快速向前；背景中，左侧岩壁与积雪交替闪过，右侧护栏沿公路延伸，远处连绵雪山和飘着粉白云彩的天空构成开阔自然景观；镜头以中景跟拍视角，平稳跟随滑板者的滑行轨迹，整体画面为实际外景拍摄，自然光线柔和地笼罩场景，呈现出运动过程中与壮美自然风光融合的视觉效果。' 
    },
    {
        label: 'Action', 
        title: 'Dynamic Action Sequences', 
        url: 'assets/videos/t2v/motion-03.mp4', 
        // prompt: '夜晚的城市道路上，多辆汽车在车道行驶，右侧有一辆橙色出租车；镜头以跟随行驶的视角向前推进，呈现出道路两旁林立的高楼、亮着光的路灯与建筑、绿化带里的树木和花丛，空中还悬着一轮明月，车辆在标线清晰的路面有序前行，灯光以夜景下的路灯和建筑灯光为主。' 
    },
    {
        label: 'Action', 
        title: 'Dynamic Action Sequences', 
        url: 'assets/videos/t2v/motion-04.mp4', 
        // prompt: '一位长发的年轻女性穿着白色无袖上衣，坐在装饰有浅色护墙板的房间里，她手持一瓶香水，边说话边转动香水瓶向镜头展示瓶身；画面右侧有一盏亮着暖光的台灯和叠放的书籍，左侧可见毛绒靠垫，整体采用中景固定镜头、平拍角度拍摄，灯光为暖调室内光，呈现实际拍摄的真人画面风格。' 
    },
    {
        label: 'Action', 
        title: 'Dynamic Action Sequences', 
        url: 'assets/videos/t2v/motion-05.mp4', 
        // prompt: 'realistic photography, A mukbang influencer sitting at a table piled high with a variety of sushi, sashimi, and tempura, ready to indulge in the spread. They begin by dipping a piece of sushi into soy sauce, slowly bringing it to their mouth and taking a big bite. The satisfying crunch of the tempura contrasts with the soft, delicate fish of the sushi, and each bite is followed by a delighted expression. As they eat, they interact with their followers, responding to comments and laughing at jokes, making the experience feel like a cozy, virtual dinner party. The gentle clink of chopsticks and the sound of eating create an oddly soothing atmosphere.' 
    },
    {
        label: 'Action', 
        title: 'Dynamic Action Sequences', 
        url: 'assets/videos/t2v/motion-06.mp4', 
        // prompt: 'realistic photography, In a sunlit dance studio with high ceilings and expansive mirrors, a graceful ballerina performs a solo routine. Her lithe body moves with fluid precision as she executes a perfect pirouette, her tutu flaring out around her. Her arms arc elegantly above her head, fingers extended in delicate curves. The dancer\'s face is a picture of serene concentration, her eyes focused on a distant point. Sunlight streams through tall windows, casting long shadows across the polished wooden floor and highlighting the dust motes that swirl in her wake. The room is silent save for the soft whisper of her pointe shoes as they brush against the floor, emphasizing the ethereal quality of her movements.' 
    },

];

const videometas_t2v_surreal: VideoMeta[] = [
    {
        label: 'Action', 
        title: 'Dynamic Action Sequences', 
        url: 'assets/videos/t2v/surreal-01.mp4', 
        // prompt: '视频采用航拍视角展现自然峡谷景观，镜头快速穿梭于两侧覆盖苔藓的陡峭岩壁之间，过程中伴随旋转、俯冲等运镜动作，清晰呈现峡谷底部流淌的河流、岩壁上的纹理，以及山间一条橙红色小径；画面以自然光呈现，天空多云，整体为真实拍摄的户外自然场景，通过动态的镜头运动展现峡谷的险峻与地貌细节。' 
    },
    {
        label: 'Action', 
        title: 'Dynamic Action Sequences', 
        url: 'assets/videos/t2v/surreal-02.mp4', 
        // prompt: '一个身着浅灰色外套、深色裤子并佩戴手套的人，在两侧有积雪覆盖山体与护栏的开阔公路上高速滑行长板；此人全程保持弯腰俯身的姿势，借助长板快速向前；背景中，左侧岩壁与积雪交替闪过，右侧护栏沿公路延伸，远处连绵雪山和飘着粉白云彩的天空构成开阔自然景观；镜头以中景跟拍视角，平稳跟随滑板者的滑行轨迹，整体画面为实际外景拍摄，自然光线柔和地笼罩场景，呈现出运动过程中与壮美自然风光融合的视觉效果。' 
    },
    {
        label: 'Action', 
        title: 'Dynamic Action Sequences', 
        url: 'assets/videos/t2v/surreal-03.mp4', 
        // prompt: '夜晚的城市道路上，多辆汽车在车道行驶，右侧有一辆橙色出租车；镜头以跟随行驶的视角向前推进，呈现出道路两旁林立的高楼、亮着光的路灯与建筑、绿化带里的树木和花丛，空中还悬着一轮明月，车辆在标线清晰的路面有序前行，灯光以夜景下的路灯和建筑灯光为主。' 
    },
    {
        label: 'Action', 
        title: 'Dynamic Action Sequences', 
        url: 'assets/videos/t2v/surreal-04.mp4', 
        // prompt: '一位长发的年轻女性穿着白色无袖上衣，坐在装饰有浅色护墙板的房间里，她手持一瓶香水，边说话边转动香水瓶向镜头展示瓶身；画面右侧有一盏亮着暖光的台灯和叠放的书籍，左侧可见毛绒靠垫，整体采用中景固定镜头、平拍角度拍摄，灯光为暖调室内光，呈现实际拍摄的真人画面风格。' 
    },
    {
        label: 'Action', 
        title: 'Dynamic Action Sequences', 
        url: 'assets/videos/t2v/surreal-05.mp4', 
        // prompt: 'realistic photography, A mukbang influencer sitting at a table piled high with a variety of sushi, sashimi, and tempura, ready to indulge in the spread. They begin by dipping a piece of sushi into soy sauce, slowly bringing it to their mouth and taking a big bite. The satisfying crunch of the tempura contrasts with the soft, delicate fish of the sushi, and each bite is followed by a delighted expression. As they eat, they interact with their followers, responding to comments and laughing at jokes, making the experience feel like a cozy, virtual dinner party. The gentle clink of chopsticks and the sound of eating create an oddly soothing atmosphere.' 
    },
    {
        label: 'Action', 
        title: 'Dynamic Action Sequences', 
        url: 'assets/videos/t2v/surreal-06.mp4', 
        // prompt: 'realistic photography, In a sunlit dance studio with high ceilings and expansive mirrors, a graceful ballerina performs a solo routine. Her lithe body moves with fluid precision as she executes a perfect pirouette, her tutu flaring out around her. Her arms arc elegantly above her head, fingers extended in delicate curves. The dancer\'s face is a picture of serene concentration, her eyes focused on a distant point. Sunlight streams through tall windows, casting long shadows across the polished wooden floor and highlighting the dust motes that swirl in her wake. The room is silent save for the soft whisper of her pointe shoes as they brush against the floor, emphasizing the ethereal quality of her movements.' 
    },

];

// const videometas_t2v_style: VideoMeta[] = [
//     {
//         label: 'Action', 
//         title: 'Dynamic Action Sequences', 
//         url: 'https://s3plus-bj02.sankuai.com/v3d-personal/liangshijun/demo-website/demo-t2v/style/0000-00-realistic%20photography%20style%2C%20a%20young%20student%2C%20arou.mp4', 
//         prompt: '视频采用航拍视角展现自然峡谷景观，镜头快速穿梭于两侧覆盖苔藓的陡峭岩壁之间，过程中伴随旋转、俯冲等运镜动作，清晰呈现峡谷底部流淌的河流、岩壁上的纹理，以及山间一条橙红色小径；画面以自然光呈现，天空多云，整体为真实拍摄的户外自然场景，通过动态的镜头运动展现峡谷的险峻与地貌细节。' 
//     },
//     {
//         label: 'Action', 
//         title: 'Dynamic Action Sequences', 
//         url: 'https://s3plus-bj02.sankuai.com/v3d-personal/liangshijun/demo-website/demo-t2v/style/0001-00-3d%20cartoon%20style%2C%20a%20young%20student%2C%20around%2018%20years.mp4', 
//         prompt: '一个身着浅灰色外套、深色裤子并佩戴手套的人，在两侧有积雪覆盖山体与护栏的开阔公路上高速滑行长板；此人全程保持弯腰俯身的姿势，借助长板快速向前；背景中，左侧岩壁与积雪交替闪过，右侧护栏沿公路延伸，远处连绵雪山和飘着粉白云彩的天空构成开阔自然景观；镜头以中景跟拍视角，平稳跟随滑板者的滑行轨迹，整体画面为实际外景拍摄，自然光线柔和地笼罩场景，呈现出运动过程中与壮美自然风光融合的视觉效果。' 
//     },
//     {
//         label: 'Action', 
//         title: 'Dynamic Action Sequences', 
//         url: 'https://s3plus-bj02.sankuai.com/v3d-personal/liangshijun/demo-website/demo-t2v/style/0003-00-anime%20style%2Ca%20young%20student%2C%20around%2018%20years%20old%2C%20.mp4', 
//         prompt: '夜晚的城市道路上，多辆汽车在车道行驶，右侧有一辆橙色出租车；镜头以跟随行驶的视角向前推进，呈现出道路两旁林立的高楼、亮着光的路灯与建筑、绿化带里的树木和花丛，空中还悬着一轮明月，车辆在标线清晰的路面有序前行，灯光以夜景下的路灯和建筑灯光为主。' 
//     },
//     {
//         label: 'Action', 
//         title: 'Dynamic Action Sequences', 
//         url: 'https://s3plus-bj02.sankuai.com/v3d-personal/liangshijun/demo-website/demo-t2v/style/0014-00-camera%20pans%20left%2C%20an%20ultra-low%20angle%2C%20wide%2C%20mid-cl.mp4', 
//         prompt: '一位长发的年轻女性穿着白色无袖上衣，坐在装饰有浅色护墙板的房间里，她手持一瓶香水，边说话边转动香水瓶向镜头展示瓶身；画面右侧有一盏亮着暖光的台灯和叠放的书籍，左侧可见毛绒靠垫，整体采用中景固定镜头、平拍角度拍摄，灯光为暖调室内光，呈现实际拍摄的真人画面风格。' 
//     },
//     {
//         label: 'Action', 
//         title: 'Dynamic Action Sequences', 
//         url: 'https://s3plus-bj02.sankuai.com/v3d-personal/liangshijun/demo-website/demo-t2v/style/0021-03-3d%20cartoon%20style%2C%20an%20ultra-low%20angle%2C%20wide%2C%20mid-cl.mp4', 
//         prompt: 'realistic photography, A mukbang influencer sitting at a table piled high with a variety of sushi, sashimi, and tempura, ready to indulge in the spread. They begin by dipping a piece of sushi into soy sauce, slowly bringing it to their mouth and taking a big bite. The satisfying crunch of the tempura contrasts with the soft, delicate fish of the sushi, and each bite is followed by a delighted expression. As they eat, they interact with their followers, responding to comments and laughing at jokes, making the experience feel like a cozy, virtual dinner party. The gentle clink of chopsticks and the sound of eating create an oddly soothing atmosphere.' 
//     },
//     {
//         label: 'Action', 
//         title: 'Dynamic Action Sequences', 
//         url: 'https://s3plus-bj02.sankuai.com/v3d-personal/liangshijun/demo-website/demo-t2v/style/0023-00-anime%20style%2Can%20ultra-low%20angle%2C%20wide%2C%20mid-close-up.mp4', 
//         prompt: 'realistic photography, In a sunlit dance studio with high ceilings and expansive mirrors, a graceful ballerina performs a solo routine. Her lithe body moves with fluid precision as she executes a perfect pirouette, her tutu flaring out around her. Her arms arc elegantly above her head, fingers extended in delicate curves. The dancer\'s face is a picture of serene concentration, her eyes focused on a distant point. Sunlight streams through tall windows, casting long shadows across the polished wooden floor and highlighting the dust motes that swirl in her wake. The room is silent save for the soft whisper of her pointe shoes as they brush against the floor, emphasizing the ethereal quality of her movements.' 
//     },

// ];

const videometas_t2v_style: VideoMeta[] = [
    {
        label: 'Action', 
        title: 'Dynamic Action Sequences', 
        url: 'assets/videos/t2v/style-01.mp4', 
        // prompt: '视频采用航拍视角展现自然峡谷景观，镜头快速穿梭于两侧覆盖苔藓的陡峭岩壁之间，过程中伴随旋转、俯冲等运镜动作，清晰呈现峡谷底部流淌的河流、岩壁上的纹理，以及山间一条橙红色小径；画面以自然光呈现，天空多云，整体为真实拍摄的户外自然场景，通过动态的镜头运动展现峡谷的险峻与地貌细节。' 
    },
    {
        label: 'Action', 
        title: 'Dynamic Action Sequences', 
        url: 'assets/videos/t2v/style-02.mp4', 
        // prompt: '一个身着浅灰色外套、深色裤子并佩戴手套的人，在两侧有积雪覆盖山体与护栏的开阔公路上高速滑行长板；此人全程保持弯腰俯身的姿势，借助长板快速向前；背景中，左侧岩壁与积雪交替闪过，右侧护栏沿公路延伸，远处连绵雪山和飘着粉白云彩的天空构成开阔自然景观；镜头以中景跟拍视角，平稳跟随滑板者的滑行轨迹，整体画面为实际外景拍摄，自然光线柔和地笼罩场景，呈现出运动过程中与壮美自然风光融合的视觉效果。' 
    },

];

const videometas_i2v: VideoMeta[] = [
    {
        label: 'Action', 
        title: 'Dynamic Action Sequences', 
        url: 'assets/videos/i2v/1-1.mp4', 
    },
    {
        label: 'Action', 
        title: 'Dynamic Action Sequences', 
        url: 'assets/videos/i2v/1-2.mp4',  
    },
    {
        label: 'Action', 
        title: 'Dynamic Action Sequences', 
        url: 'assets/videos/i2v/1-3.mp4',  
    },
    {
        label: 'Action', 
        title: 'Dynamic Action Sequences', 
        url: 'assets/videos/i2v/1-4.mp4', 
    },
    {
        label: 'Action', 
        title: 'Dynamic Action Sequences', 
        url: 'assets/videos/i2v/1-5.mp4',  
    },
    {
        label: 'Action', 
        title: 'Dynamic Action Sequences', 
        url: 'assets/videos/i2v/1-6.mp4',  
    },


];

const videometas_i2v_1: VideoMeta[] = [

    {
        label: 'Action', 
        title: 'Dynamic Action Sequences', 
        url: 'assets/videos/i2v/2-1.mp4', 
    },
    {
        label: 'Action', 
        title: 'Dynamic Action Sequences', 
        url: 'assets/videos/i2v/2-2.mp4', 
    },
    {
        label: 'Action', 
        title: 'Dynamic Action Sequences', 
        url: 'assets/videos/i2v/2-3.mp4', 
    },
    {
        label: 'Action', 
        title: 'Dynamic Action Sequences', 
        url: 'assets/videos/i2v/2-4.mp4', 
    },

];

const videometas_i2v_2: VideoMeta[] = [

    {
        label: 'Action', 
        title: 'Dynamic Action Sequences', 
        url: 'assets/videos/i2v/3-1.mp4', 
    },
    {
        label: 'Action', 
        title: 'Dynamic Action Sequences', 
        url: 'assets/videos/i2v/3-2.mp4', 
    },
    {
        label: 'Action', 
        title: 'Dynamic Action Sequences', 
        url: 'assets/videos/i2v/3-3.mp4', 
    },

];

const videometas_long_0: VideoMeta[] = [

    {
        label: 'Action', 
        title: 'Dynamic Action Sequences', 
        url: 'assets/videos/long/1-1.mp4', 
    },
    {
        label: 'Action', 
        title: 'Dynamic Action Sequences', 
        url: 'assets/videos/long/1-2.mp4', 
    },
    {
        label: 'Action', 
        title: 'Dynamic Action Sequences', 
        url: 'assets/videos/long/1-3.mp4', 
    },
    {
        label: 'Action', 
        title: 'Dynamic Action Sequences', 
        url: 'assets/videos/long/1-4.mp4', 
    },
    {
        label: 'Action', 
        title: 'Dynamic Action Sequences', 
        url: 'assets/videos/long/1-5.mp4', 
    },
    {
        label: 'Action', 
        title: 'Dynamic Action Sequences', 
        url: 'assets/videos/long/1-6.mp4', 
    },
];

const videometas_long_1: VideoMeta[] = [

    {
        label: 'Action', 
        title: 'Dynamic Action Sequences', 
        url: 'assets/videos/long/2-1.mp4', 
    },
    {
        label: 'Action', 
        title: 'Dynamic Action Sequences', 
        url: 'assets/videos/long/2-2.mp4', 
    },
    {
        label: 'Action', 
        title: 'Dynamic Action Sequences', 
        url: 'assets/videos/long/2-3.mp4', 
    },
    {
        label: 'Action', 
        title: 'Dynamic Action Sequences', 
        url: 'assets/videos/long/2-4.mp4', 
    },
    {
        label: 'Action', 
        title: 'Dynamic Action Sequences', 
        url: 'assets/videos/long/2-5.mp4', 
    },
    {
        label: 'Action', 
        title: 'Dynamic Action Sequences', 
        url: 'assets/videos/long/2-6.mp4', 
    },
];

const videometas_long_2: VideoMeta[] = [

    {
        label: 'Action', 
        title: 'Dynamic Action Sequences', 
        url: 'assets/videos/long/3-1.mp4', 
    },
    {
        label: 'Action', 
        title: 'Dynamic Action Sequences', 
        url: 'assets/videos/long/3-2.mp4', 
    },
    {
        label: 'Action', 
        title: 'Dynamic Action Sequences', 
        url: 'assets/videos/long/3-3.mp4', 
    },
    {
        label: 'Action', 
        title: 'Dynamic Action Sequences', 
        url: 'assets/videos/long/3-4.mp4', 
    },
    {
        label: 'Action', 
        title: 'Dynamic Action Sequences', 
        url: 'assets/videos/long/3-5.mp4', 
    },
    {
        label: 'Action', 
        title: 'Dynamic Action Sequences', 
        url: 'assets/videos/long/3-6.mp4', 
    },
];

const videometas_method_0: VideoMeta[] = [

    {
        label: 'Action', 
        title: 'Dynamic Action Sequences', 
        url: 'assets/images/method-1.png', 
        prompt: 'to their mouth and taking a big bite. The satisfying crunch of the tempura contrasts with the soft, delicate fish of the sushi, and each bite is followed by a deligh'
    },
];

const videometas_method_1: VideoMeta[] = [

    {
        label: 'Action', 
        title: 'Dynamic Action Sequences', 
        url: 'assets/images/method-2.png', 
        prompt: 'to their mouth and taking a big bite. The satisfying crunch of the tempura contrasts with the soft, delicate fish of the sushi, and each bite is followed by a deligh'
    },
];

const videometas_method_2: VideoMeta[] = [

    {
        label: 'Action', 
        title: 'Dynamic Action Sequences', 
        url: 'assets/images/method-3.png', 
        prompt: 'to their mouth and taking a big bite. The satisfying crunch of the tempura contrasts with the soft, delicate fish of the sushi, and each bite is followed by a deligh'
    },
];

const videometas_method_3: VideoMeta[] = [

    {
        label: 'Action', 
        title: 'Dynamic Action Sequences', 
        url: 'assets/images/method-4.png', 
        prompt: 'to their mouth and taking a big bite. The satisfying crunch of the tempura contrasts with the soft, delicate fish of the sushi, and each bite is followed by a deligh'
    },
];

const videometas_hs: VideoMeta[] = [
    {
        label: 'Action', 
        title: 'Dynamic Action Sequences', 
        url: 'https://s3plus-bj02.sankuai.com/v3d-personal/liangshijun/demo-website/long/0008-%E8%A7%86%E9%A2%91%E9%87%87%E7%94%A8%E8%88%AA%E6%8B%8D%E8%A7%86%E8%A7%92%E5%B1%95%E7%8E%B0%E8%87%AA%E7%84%B6%E5%B3%A1%E8%B0%B7%E6%99%AF%E8%A7%82%EF%BC%8C%E9%95%9C%E5%A4%B4%E5%BF%AB%E9%80%9F%E7%A9%BF%E6%A2%AD%E4%BA%8E%E4%B8%A4%E4%BE%A7%E8%A6%86%E7%9B%96%E8%8B%94%E8%97%93%E7%9A%84%E9%99%A1%E5%B3%AD%E5%B2%A9%E5%A3%81%E4%B9%8B%E9%97%B4%EF%BC%8C%E8%BF%87%E7%A8%8B%E4%B8%AD%E4%BC%B4%E9%9A%8F%E6%97%8B%E8%BD%AC%E3%80%81%E4%BF%AF%E5%86%B2%E7%AD%89%E8%BF%90.mp4', 
        prompt: '视频采用航拍视角展现自然峡谷景观，镜头快速穿梭于两侧覆盖苔藓的陡峭岩壁之间，过程中伴随旋转、俯冲等运镜动作，清晰呈现峡谷底部流淌的河流、岩壁上的纹理，以及山间一条橙红色小径；画面以自然光呈现，天空多云，整体为真实拍摄的户外自然场景，通过动态的镜头运动展现峡谷的险峻与地貌细节。' 
    },
    {
        label: 'Action', 
        title: 'Dynamic Action Sequences', 
        url: 'https://s3plus-bj02.sankuai.com/v3d-personal/liangshijun/demo-website/long/0013-%E4%B8%80%E4%B8%AA%E8%BA%AB%E7%9D%80%E6%B5%85%E7%81%B0%E8%89%B2%E5%A4%96%E5%A5%97%E3%80%81%E6%B7%B1%E8%89%B2%E8%A3%A4%E5%AD%90%E5%B9%B6%E4%BD%A9%E6%88%B4%E6%89%8B%E5%A5%97%E7%9A%84%E4%BA%BA%EF%BC%8C%E5%9C%A8%E4%B8%A4%E4%BE%A7%E6%9C%89%E7%A7%AF%E9%9B%AA%E8%A6%86%E7%9B%96%E5%B1%B1%E4%BD%93%E4%B8%8E%E6%8A%A4%E6%A0%8F%E7%9A%84%E5%BC%80%E9%98%94%E5%85%AC%E8%B7%AF%E4%B8%8A%E9%AB%98%E9%80%9F%E6%BB%91%E8%A1%8C%E9%95%BF%E6%9D%BF%EF%BC%9B%E6%AD%A4%E4%BA%BA.mp4', 
        prompt: '一个身着浅灰色外套、深色裤子并佩戴手套的人，在两侧有积雪覆盖山体与护栏的开阔公路上高速滑行长板；此人全程保持弯腰俯身的姿势，借助长板快速向前；背景中，左侧岩壁与积雪交替闪过，右侧护栏沿公路延伸，远处连绵雪山和飘着粉白云彩的天空构成开阔自然景观；镜头以中景跟拍视角，平稳跟随滑板者的滑行轨迹，整体画面为实际外景拍摄，自然光线柔和地笼罩场景，呈现出运动过程中与壮美自然风光融合的视觉效果。' 
    },
    {
        label: 'Action', 
        title: 'Dynamic Action Sequences', 
        url: 'https://s3plus-bj02.sankuai.com/v3d-personal/liangshijun/demo-website/long/0000-%E5%A4%9C%E6%99%9A%E7%9A%84%E5%9F%8E%E5%B8%82%E9%81%93%E8%B7%AF%E4%B8%8A%EF%BC%8C%E5%A4%9A%E8%BE%86%E6%B1%BD%E8%BD%A6%E5%9C%A8%E8%BD%A6%E9%81%93%E8%A1%8C%E9%A9%B6%EF%BC%8C%E5%8F%B3%E4%BE%A7%E6%9C%89%E4%B8%80%E8%BE%86%E6%A9%99%E8%89%B2%E5%87%BA%E7%A7%9F%E8%BD%A6%EF%BC%9B%E9%95%9C%E5%A4%B4%E4%BB%A5%E8%B7%9F%E9%9A%8F%E8%A1%8C%E9%A9%B6%E7%9A%84%E8%A7%86%E8%A7%92%E5%90%91%E5%89%8D%E6%8E%A8%E8%BF%9B%EF%BC%8C%E5%91%88%E7%8E%B0%E5%87%BA%E9%81%93%E8%B7%AF.mp4', 
        prompt: '夜晚的城市道路上，多辆汽车在车道行驶，右侧有一辆橙色出租车；镜头以跟随行驶的视角向前推进，呈现出道路两旁林立的高楼、亮着光的路灯与建筑、绿化带里的树木和花丛，空中还悬着一轮明月，车辆在标线清晰的路面有序前行，灯光以夜景下的路灯和建筑灯光为主。' 
    },
    {
        label: 'Action', 
        title: 'Dynamic Action Sequences', 
        url: 'https://s3plus-bj02.sankuai.com/v3d-personal/liangshijun/demo-website/long/0022-%E4%B8%80%E4%BD%8D%E9%95%BF%E5%8F%91%E7%9A%84%E5%B9%B4%E8%BD%BB%E5%A5%B3%E6%80%A7%E7%A9%BF%E7%9D%80%E7%99%BD%E8%89%B2%E6%97%A0%E8%A2%96%E4%B8%8A%E8%A1%A3%EF%BC%8C%E5%9D%90%E5%9C%A8%E8%A3%85%E9%A5%B0%E6%9C%89%E6%B5%85%E8%89%B2%E6%8A%A4%E5%A2%99%E6%9D%BF%E7%9A%84%E6%88%BF%E9%97%B4%E9%87%8C%EF%BC%8C%E5%A5%B9%E6%89%8B%E6%8C%81%E4%B8%80%E7%93%B6%E9%A6%99%E6%B0%B4%EF%BC%8C%E8%BE%B9%E8%AF%B4%E8%AF%9D%E8%BE%B9%E8%BD%AC%E5%8A%A8%E9%A6%99%E6%B0%B4%E7%93%B6.mp4', 
        prompt: '一位长发的年轻女性穿着白色无袖上衣，坐在装饰有浅色护墙板的房间里，她手持一瓶香水，边说话边转动香水瓶向镜头展示瓶身；画面右侧有一盏亮着暖光的台灯和叠放的书籍，左侧可见毛绒靠垫，整体采用中景固定镜头、平拍角度拍摄，灯光为暖调室内光，呈现实际拍摄的真人画面风格。' 
    },
    {
        label: 'Action', 
        title: 'Dynamic Action Sequences', 
        url: 'https://s3plus-bj02.sankuai.com/v3d-personal/liangshijun/demo-website/long/0015-realistic%20photography%2C%20A%20mukbang%20influencer%20sittin.mp4', 
        prompt: 'realistic photography, A mukbang influencer sitting at a table piled high with a variety of sushi, sashimi, and tempura, ready to indulge in the spread. They begin by dipping a piece of sushi into soy sauce, slowly bringing it to their mouth and taking a big bite. The satisfying crunch of the tempura contrasts with the soft, delicate fish of the sushi, and each bite is followed by a delighted expression. As they eat, they interact with their followers, responding to comments and laughing at jokes, making the experience feel like a cozy, virtual dinner party. The gentle clink of chopsticks and the sound of eating create an oddly soothing atmosphere.' 
    },
    {
        label: 'Action', 
        title: 'Dynamic Action Sequences', 
        url: 'https://s3plus-bj02.sankuai.com/v3d-personal/liangshijun/demo-website/long/0008-realistic%20photography%2C%20In%20a%20sunlit%20dance%20studio%20wi.mp4', 
        prompt: 'realistic photography, In a sunlit dance studio with high ceilings and expansive mirrors, a graceful ballerina performs a solo routine. Her lithe body moves with fluid precision as she executes a perfect pirouette, her tutu flaring out around her. Her arms arc elegantly above her head, fingers extended in delicate curves. The dancer\'s face is a picture of serene concentration, her eyes focused on a distant point. Sunlight streams through tall windows, casting long shadows across the polished wooden floor and highlighting the dust motes that swirl in her wake. The room is silent save for the soft whisper of her pointe shoes as they brush against the floor, emphasizing the ethereal quality of her movements.' 
    },

];




interface TimedPrompt {
    time: number; // 时间戳（秒）
    text: string; // 字幕文本
}
  
export interface VideoMetaMP {
//   id: string;
  url: string;
  label: string;
  title: string;
  prompts: TimedPrompt[];
}

const videometas_mp: VideoMetaMP[] = [
    { 
        label: 'Action', 
        title: 'Dynamic Action Sequences', 
        url: 'https://s3plus-bj02.sankuai.com/v3d-personal/liangshijun/demo-website/multi/0002-The%20kitchen%20has%20a%20rustic%20charm%2C%20featuring%20wooden%20c.mp4',
        prompts: [
           { time: 0, text: 'The kitchen has a rustic charm, featuring wooden cabinets and a countertop adorned with a colorful fruit bowl. A small jar of honey is placed next to a white teacup on the countertop, ready for use. A spoon is placed next to the jar of honey. A woman in a floral apron stands at the countertop, carefully pouring steaming tea from a teapot into a delicate white teacup. The tea flows smoothly, filling the cup with a rich amber color.' },
           { time: 6, text: 'The woman puts down the teapot in her hand.' },
           { time: 11, text: 'The woman picks up the spoon on the plate, dips the spoon into the honey jar and adds some honey to the teacup with the spoon.' },
           { time: 16, text: 'The woman stirs the tea gently with the spoon.' },
        ] 
    },
    { 
        label: 'Action', 
        title: 'Dynamic Action Sequences', 
        url: 'https://s3plus-bj02.sankuai.com/v3d-personal/liangshijun/demo-website/multi/0002-Camera%20zooms%20out%2C%20the%20woman%20puts%20down%20the%20knife%20in.mp4',
        prompts: [
           { time: 0, text: 'The kitchen is bright and airy, featuring white cabinets and a wooden countertop. A loaf of freshly baked bread rests on a cutting board, and a glass and a carton of milk are positioned nearby. A woman wearing a floral apron stands at the wooden countertop, skillfully slicing a golden-brown loaf of bread with a sharp knife. The bread is resting on a cutting board, and crumbs scatter around as she cuts.' },
           { time: 6, text: 'Camera zooms out, The woman puts down the knife in her hand, reaches for the carton of milk and then pours it into the glass on the table.' },
           { time: 11, text: 'The woman puts down the milk carton.' },
           { time: 16, text: 'The woman picks up the glass of milk and takes a sip.' },
        ] 
    },
    { 
        label: 'Action', 
        title: 'Dynamic Action Sequences', 
        url: 'https://s3plus-bj02.sankuai.com/v3d-personal/liangshijun/demo-website/multi/0002-The%20woman%20is%20in%20a%20bright%2C%20modern%20living%20room%20with%20.mp4',
        prompts: [
           { time: 0, text: 'The woman is in a bright, modern living room with a large window letting in natural light. She is wearing a casual white t-shirt and jeans, sitting comfortably on a plush gray sofa. A small coffee table in front of her holds a few magazines and a decorative plant. The woman is seated on the sofa, holding her smartphone in her left hand while intently looking at the screen. With her right hand, she taps on the screen, possibly scrolling or selecting an app.' },
           { time: 6, text: 'The woman positions her phone, makes a V sign, and takes a selfie.' },
           { time: 11, text: 'The woman places her phone on the sofa and stretches her arms.' },
           { time: 16, text: 'The woman walks to the window and opens it.' },
        ] 
    },
    { 
        label: 'Action', 
        title: 'Dynamic Action Sequences', 
        url: 'https://s3plus-bj02.sankuai.com/v3d-personal/liangshijun/demo-website/multi/0004-The%20bathroom%20is%20elegantly%20designed%20with%20marble%20cou.mp4',
        prompts: [
           { time: 0, text: 'The bathroom is elegantly designed with marble countertops and a large, well-lit mirror above the sink. A variety of makeup products are neatly arranged on the countertop, and a plush towel hangs on a rack nearby. The woman is wearing a stylish black dress and has her hair styled in loose waves. The woman stands in front of the large mirror, gently adjusting its angle to get a better view. She appears focused, ensuring the mirror reflects her image perfectly.' },
           { time: 6, text: 'The woman turns on the faucet and starts washing her hands.' },
           { time: 11, text: 'The woman turns off the faucet and picks up the towel hanging on the wall.' },
           { time: 16, text: 'The woman dries her hands while looking herself in the mirror.' },
        ] 
    },
    { 
        label: 'Action', 
        title: 'Dynamic Action Sequences', 
        url: 'https://s3plus-bj02.sankuai.com/v3d-personal/liangshijun/demo-website/multi/0013-A%20young%20man%20wearing%20a%20gray%20t-shirt%20and%20blue%20jeans%20.mp4',
        prompts: [
           { time: 0, text: 'A young man wearing a gray t-shirt and blue jeans is seated at a modern wooden desk in a well-lit room. The desk holds a sleek laptop, a pair of black headphones resting beside it, and a small potted plant in the corner. The walls are painted a soft white, creating a bright and airy atmosphere. The young man is focused as he types on the laptop keyboard with both hands, his fingers moving swiftly over the keys. The screen displays a document he is working on.' },
           { time: 6, text: 'The man reaches out to the headphones and put them on.' },
           { time: 11, text: 'The man closes the laptop.' },
           { time: 16, text: 'The man stands up from the chair and moves away from the desk.' },
        ] 
    },
    { 
        label: 'Action', 
        title: 'Dynamic Action Sequences', 
        url: 'https://s3plus-bj02.sankuai.com/v3d-personal/liangshijun/demo-website/multi/0015-The%20woman%2C%20dressed%20in%20a%20vibrant%20red%20blouse%2C%20stands.mp4',
        prompts: [
           { time: 0, text: 'The woman, dressed in a vibrant red blouse, stands in a bright, airy living room with large windows letting in natural light. The room is decorated with potted plants and a comfortable sofa in the background. She has shoulder-length hair and a warm smile, creating an inviting atmosphere. The woman raises her right hand and waves enthusiastically, her smile widening as she greets someone off-screen. The gesture is friendly and inviting, reflecting her cheerful demeanor.' },
           { time: 6, text: 'The woman begins to talk, gesturing with both hands.' },
           { time: 11, text: 'The woman brings her hands together to form a heart shape, holding it close to her chest.' },
           { time: 16, text: 'The woman brings her right hand to her lips and blows a kiss towards the camera.' },
        ] 
    },

];

const videometas_mp_0: VideoMetaMP[] = [
    { 
        label: 'Action', 
        title: 'Dynamic Action Sequences', 
        url: 'assets/videos/interaction/1-1.mp4',
        prompts: [
            { time: 0, text: 'The kitchen is bright and airy, featuring white cabinets and a wooden countertop. A loaf of freshly baked bread rests on a cutting board, and a glass and a carton of milk are positioned nearby. A woman wearing a floral apron stands at the wooden countertop, skillfully slicing a golden-brown loaf of bread with a sharp knife. The bread is resting on a cutting board, and crumbs scatter around as she cuts.' },
            { time: 6, text: 'Camera zooms out, The woman puts down the knife in her hand, reaches for the carton of milk and then pours it into the glass on the table.' },
            { time: 11, text: 'The woman puts down the milk carton.' },
            { time: 16, text: 'The woman picks up the glass of milk and takes a sip.' },
         ] 
        
    },
    { 
        label: 'Action', 
        title: 'Dynamic Action Sequences', 
        url: 'assets/videos/interaction/1-2.mp4',
        prompts: [
            { time: 0, text: 'The kitchen has a rustic charm, featuring wooden cabinets and a countertop adorned with a colorful fruit bowl. A small jar of honey is placed next to a white teacup on the countertop, ready for use. A spoon is placed next to the jar of honey. A woman in a floral apron stands at the countertop, carefully pouring steaming tea from a teapot into a delicate white teacup. The tea flows smoothly, filling the cup with a rich amber color.' },
            { time: 6, text: 'The woman puts down the teapot in her hand.' },
            { time: 11, text: 'The woman picks up the spoon on the plate, dips the spoon into the honey jar and adds some honey to the teacup with the spoon.' },
            { time: 16, text: 'The woman stirs the tea gently with the spoon.' },
         ] 
    },
];

const videometas_mp_1: VideoMetaMP[] = [

    { 
        label: 'Action', 
        title: 'Dynamic Action Sequences', 
        url: 'assets/videos/interaction/2-1.mp4',
        prompts: [
           { time: 0, text: 'The woman is in a bright, modern living room with a large window letting in natural light. She is wearing a casual white t-shirt and jeans, sitting comfortably on a plush gray sofa. A small coffee table in front of her holds a few magazines and a decorative plant. The woman is seated on the sofa, holding her smartphone in her left hand while intently looking at the screen. With her right hand, she taps on the screen, possibly scrolling or selecting an app.' },
           { time: 6, text: 'The woman positions her phone, makes a V sign, and takes a selfie.' },
           { time: 11, text: 'The woman places her phone on the sofa and stretches her arms.' },
           { time: 16, text: 'The woman walks to the window and opens it.' },
        ] 
    },
    { 
        label: 'Action', 
        title: 'Dynamic Action Sequences', 
        url: 'assets/videos/interaction/2-2.mp4',
        prompts: [
           { time: 0, text: 'The bathroom is elegantly designed with marble countertops and a large, well-lit mirror above the sink. A variety of makeup products are neatly arranged on the countertop, and a plush towel hangs on a rack nearby. The woman is wearing a stylish black dress and has her hair styled in loose waves. The woman stands in front of the large mirror, gently adjusting its angle to get a better view. She appears focused, ensuring the mirror reflects her image perfectly.' },
           { time: 6, text: 'The woman turns on the faucet and starts washing her hands.' },
           { time: 11, text: 'The woman picks up the towel hanging on the wall.' },
           { time: 16, text: 'The woman dries her hands while looking herself in the mirror.' },
        ] 
    },

];

const videometas_mp_2: VideoMetaMP[] = [
    { 
        label: 'Action', 
        title: 'Dynamic Action Sequences', 
        url: 'assets/videos/interaction/3-1.mp4',
        prompts: [
           { time: 0, text: 'A young man wearing a gray t-shirt and blue jeans is seated at a modern wooden desk in a well-lit room. The desk holds a sleek laptop, a pair of black headphones resting beside it, and a small potted plant in the corner. The walls are painted a soft white, creating a bright and airy atmosphere. The young man is focused as he types on the laptop keyboard with both hands, his fingers moving swiftly over the keys. The screen displays a document he is working on.' },
           { time: 6, text: 'The man reaches out to the headphones and put them on.' },
           { time: 11, text: 'The man closes the laptop.' },
           { time: 16, text: 'The man stands up from the chair and moves away from the desk.' },
        ] 
    },
    { 
        label: 'Action', 
        title: 'Dynamic Action Sequences', 
        url: 'assets/videos/interaction/3-2.mp4',
        prompts: [
           { time: 0, text: 'The woman, dressed in a vibrant red blouse, stands in a bright, airy living room with large windows letting in natural light. The room is decorated with potted plants and a comfortable sofa in the background. She has shoulder-length hair and a warm smile, creating an inviting atmosphere. The woman raises her right hand and waves enthusiastically, her smile widening as she greets someone off-screen. The gesture is friendly and inviting, reflecting her cheerful demeanor.' },
           { time: 6, text: 'The woman begins to talk, gesturing with both hands.' },
           { time: 11, text: 'The woman brings her hands together to form a heart shape, holding it close to her chest.' },
           { time: 16, text: 'The woman brings her right hand to her lips and blows a kiss towards the camera.' },
        ] 
    },

];
  
  // 导出所有内容
export { 
    videometas, videometas_hs, videometas_mp, 
    videometas_t2v_motion, videometas_t2v_surreal, videometas_t2v_style, 
    videometas_i2v, videometas_i2v_1, videometas_i2v_2,
    videometas_long_0, videometas_long_1, videometas_long_2,
    videometas_mp_0, videometas_mp_1, videometas_mp_2,
    videometas_method_0, videometas_method_1, videometas_method_2, videometas_method_3
};
//   export default videometas; // 默认导出第一个数组