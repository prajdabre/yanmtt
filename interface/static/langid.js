// langid.js is a direct port of langid.py to JavaScript.
// For license conditions, see the LICENSE file in the same repository.
// Marco Lui <saffsd@gmail.com>, July 2014

"use strict";

var langid = (function() {
  var my = {};

  function base64ToArray(encStr, arraytype) {
    var decStr = atob(encStr)
    var buf = new ArrayBuffer(decStr.length);
    var bufWrite = new Uint8Array(buf);
    for (var i=0, bufSize=bufWrite.length; i<bufSize; i++){
      bufWrite[i] = decStr.charCodeAt(i);
    }
    var bufView = new arraytype(buf);
    return bufView;
  }

  // unpack the model. the _xxx packed version is externally supplied.
  var tk_nextmove = base64ToArray(_tk_nextmove, Uint16Array);
  var tk_output_packed = base64ToArray(_tk_output, Uint16Array);
  var nb_pc = base64ToArray(_nb_pc, Float64Array);
  var nb_ptc = base64ToArray(_nb_ptc, Float64Array);
  var nb_classes = _nb_classes

  // unpack tk_output
  var tk_output = {};
  var limit = tk_output_packed[0];
  for (var i=0, j=1; i < limit; i++) {
    var s = tk_output_packed[j];
    var c = tk_output_packed[j+1];
    var arr = tk_output_packed.subarray(j+2,j+2+c);

    tk_output[s] = arr;
    j += 2+c;
  }

  // calculate some properties of the model
  var num_langs = nb_classes.length;
  var num_features = nb_ptc.length / num_langs;
  var num_states = tk_nextmove.length / 256;

  console.log("unpacked a langid model: " + num_langs + " langs, " + num_features + " feats, " + num_states + " states.");

  my.textToFv = function(str){
    // convert raw input text to a vector of transitions.

    // The model in langid.js operates at a byte level, and most
    // of the training data used was UTF8, so we need to first encode 
    // the string in UTF8 before processing.
    var enc = unescape(encodeURIComponent(str));

    var sv = new Uint32Array(num_states);
    var s = 0; // start at state 0;
    for (var i=0, l=enc.length; i<l; i++){
      var c = enc.charCodeAt(i);
      s = tk_nextmove[(s<<8)+c];
      sv[s] += 1;
    }

    // convert the transitions into feature counts
    var fv = new Uint32Array(num_features);
    for (var i=0, l=num_states; i<l; i++){
      if ((sv[i] > 0) && (i in tk_output)){
        var states = tk_output[i];
        for (var j=0, m=states.length; j<m; j++){
          fv[states[j]] += sv[i]; // increment corresponding features 
        }
      }
    }

    return fv;
  }

  my.fvToLogprob = function(fv){
    // rank languages based on an input fv
    var logprob = new Float64Array(nb_pc);
    for (var i = 0; i < num_features; i++){
      if (fv[i] > 0){
        for (var j=0; j < num_langs; j++){
          logprob[j] += fv[i] * nb_ptc[i*num_langs + j];
        }
      }
    }
    return logprob;
  }

  my.logprobToPred = function(logprob){
    var _i = 0;

    for (var i=1;i<num_langs;i++){
      if (logprob[_i] < logprob[i]) _i = i;
    }
    console.log('pred: '+_i+ ' lang: '+ nb_classes[_i] + ' logprob: ' + logprob[_i]);
    return nb_classes[_i];
  }

  my.logprobToRank= function(logprob){
    var preds = [];
    for (var i=0;i<num_langs;i++) preds.push({"lang":nb_classes[i], "logprob":logprob[i]});
    preds.sort(function(a,b){return b["logprob"]-a["logprob"];});

    return preds;
  }

  my.identify = function(str){
    var fv = my.textToFv(str);
    var lp = my.fvToLogprob(fv);
    var pred = my.logprobToPred(lp);
    return pred;
  }

  my.rank = function(str){
    var fv = my.textToFv(str);
    var lp = my.fvToLogprob(fv);
    var rank = my.logprobToRank(lp);
    return rank;
  }

  return my;
  
})();
