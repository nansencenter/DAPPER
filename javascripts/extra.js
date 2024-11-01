// https://squidfunk.github.io/mkdocs-material/customization/#additional-javascript
console.log("This message appears on every page")

// Tag converted notebooks (to facilitate custom styling)
document.addEventListener("DOMContentLoaded", function() {
  if (document.querySelector('.jp-Notebook')) {
    document.body.classList.add('document-is-notebook');
  }
});
