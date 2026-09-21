(function(){
  const fileInput=document.getElementById('file');
  const preview=document.getElementById('preview');
  if(fileInput&&preview){fileInput.addEventListener('change',function(){const file=this.files&&this.files[0];if(!file)return;const reader=new FileReader();reader.onload=e=>{preview.src=e.target.result;preview.style.display='block';};reader.readAsDataURL(file);});}
  const video=document.getElementById('video');
  const canvas=document.getElementById('overlay');
  const start=document.getElementById('startCamera');
  const stop=document.getElementById('stopCamera');
  const status=document.getElementById('cameraStatus');
  const fpsEl=document.getElementById('fps');
  const placeholder=document.getElementById('camera-placeholder');
  const errorEl=document.getElementById('cameraError');
  if(!video||!canvas||!start||!stop)return;
  let stream=null, timer=null, busy=false, frames=0, lastFps=performance.now();
  const ctx=canvas.getContext('2d');
  function setError(msg){errorEl.textContent=msg;errorEl.classList.toggle('hidden',!msg);}
  function resize(){if(video.videoWidth){canvas.width=video.videoWidth;canvas.height=video.videoHeight;}}
  async function analyze(){if(!stream||busy||video.readyState<2)return;busy=true;try{const temp=document.createElement('canvas');temp.width=video.videoWidth;temp.height=video.videoHeight;temp.getContext('2d').drawImage(video,0,0);const response=await fetch('/predict_frame',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({image:temp.toDataURL('image/jpeg',.75)})});const data=await response.json();ctx.clearRect(0,0,canvas.width,canvas.height);if(data.success){(data.faces||[]).forEach(face=>{ctx.strokeStyle=face.color||'#16a34a';ctx.lineWidth=Math.max(2,canvas.width/320);ctx.strokeRect(face.x,face.y,face.w,face.h);ctx.fillStyle=face.color||'#16a34a';ctx.font='bold 16px Arial';const label=face.label+' '+face.confidence+'%';const width=ctx.measureText(label).width+14;ctx.fillRect(face.x,Math.max(0,face.y-28),width,28);ctx.fillStyle='#fff';ctx.fillText(label,face.x+7,Math.max(19,face.y-9));});frames++;const now=performance.now();if(now-lastFps>1000){fpsEl.textContent=frames+' FPS';frames=0;lastFps=now;}}else{setError(data.error||'Frame processing failed.');}}catch(e){setError('Unable to reach the detection server.');}finally{busy=false;}}
  async function startCamera(){setError('');try{stream=await navigator.mediaDevices.getUserMedia({video:{facingMode:'user'},audio:false});video.srcObject=stream;await video.play();resize();placeholder.style.display='none';start.disabled=true;stop.disabled=false;status.textContent='Camera running';timer=setInterval(analyze,180);}catch(e){setError('Camera permission was denied or the camera is unavailable.');status.textContent='Camera unavailable';}}
  function stopCamera(){if(timer)clearInterval(timer);timer=null;if(stream)stream.getTracks().forEach(t=>t.stop());stream=null;video.srcObject=null;ctx.clearRect(0,0,canvas.width,canvas.height);placeholder.style.display='block';start.disabled=false;stop.disabled=true;status.textContent='Idle';fpsEl.textContent='0 FPS';}
  start.addEventListener('click',startCamera);stop.addEventListener('click',stopCamera);video.addEventListener('loadedmetadata',resize);window.addEventListener('beforeunload',stopCamera);
})();
