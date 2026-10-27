// Helper for code/00a_extract_pdf_text.sh – turns one PDF into a text file.
// Uses Apple's built-in PDFKit, so nothing needs to be installed.
// Usage: osascript -l JavaScript code/helpers/pdf2txt.js <input.pdf> <output.txt>
ObjC.import('PDFKit');

function run(argv) {
  var pdf = $.PDFDocument.alloc.initWithURL($.NSURL.fileURLWithPath(argv[0]));
  var pages = [];
  for (var i = 0; i < pdf.pageCount; i++) {
    // mark each page so we can cite page numbers while reading
    pages.push('=== PAGE ' + (i + 1) + ' ===\n' + ObjC.unwrap(pdf.pageAtIndex(i).string));
  }
  $(pages.join('\n')).writeToFileAtomicallyEncodingError(argv[1], true, $.NSUTF8StringEncoding, null);
  return pdf.pageCount + ' pages';
}
