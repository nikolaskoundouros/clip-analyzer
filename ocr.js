// Reads the text in every PNG of a folder with Apple's on-device text reader (Vision), through
// JavaScript for Automation, so no compiler is needed. Prints: name, a tab, the text.
// Usage: osascript -l JavaScript ocr.js FOLDER
ObjC.import('Foundation');
ObjC.import('Vision');
function run(argv) {
  const dir = argv[0], fm = $.NSFileManager.defaultManager;
  const names = ObjC.deepUnwrap(fm.contentsOfDirectoryAtPathError(dir, null)).filter(n => n.endsWith('.png')).sort();
  const out = [];
  for (const n of names) {
    const url = $.NSURL.fileURLWithPath(dir + '/' + n);
    const req = $.VNRecognizeTextRequest.alloc.init;
    req.recognitionLevel = 0;              // 0 = accurate
    req.usesLanguageCorrection = false;
    const handler = $.VNImageRequestHandler.alloc.initWithURLOptions(url, $.NSDictionary.dictionary);
    handler.performRequestsError($.NSArray.arrayWithObject(req), null);
    const res = req.results, parts = [];
    for (let i = 0; i < res.count; i++) {
      const c = res.objectAtIndex(i).topCandidates(1);
      if (c.count > 0) parts.push(ObjC.unwrap(c.objectAtIndex(0).string));
    }
    out.push(n + '\t' + parts.join(' '));
  }
  return out.join('\n');
}
