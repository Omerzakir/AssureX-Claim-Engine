'use strict';
const readline = require('readline');
const path = require('path');
const {predictImage} = require('./tm_predictor');
const modelDir = process.env.ASSUREX_TM_MODEL_DIR || path.resolve(__dirname, '../../models/teachable_machine');
const rl = readline.createInterface({input:process.stdin,crlfDelay:Infinity});
rl.on('line', async line => {
  try { const req=JSON.parse(line); const result=await predictImage(req.image_path,modelDir); process.stdout.write(JSON.stringify({ok:true,result})+'\n'); }
  catch(e){ process.stdout.write(JSON.stringify({ok:false,error:e.message})+'\n'); }
});
