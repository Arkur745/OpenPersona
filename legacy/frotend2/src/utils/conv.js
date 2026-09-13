const CONV_ID_KEY = 'chat_conv_id';

export function getConvId() {
  let convId = localStorage.getItem(CONV_ID_KEY);
  
  if (!convId) {
    convId = `conv_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
    localStorage.setItem(CONV_ID_KEY, convId);
  }
  
  return convId;
}

export function resetConvId() {
  localStorage.removeItem(CONV_ID_KEY);
  return getConvId();
}
