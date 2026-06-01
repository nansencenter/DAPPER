// https://squidfunk.github.io/mkdocs-material/customization/#additional-javascript
console.log("This message appears on every page")

// Tag converted notebooks (to facilitate custom styling)
document.addEventListener("DOMContentLoaded", function() {
  if (document.querySelector('.jp-Notebook')) {
    document.body.classList.add('document-is-notebook');
  }
});

// Using the document$ observable from mkdocs-material to get notified of page "reload" also if using `navigation.instant` (SSA)
// https://github.com/danielfrg/mkdocs-jupyter/issues/99#issuecomment-2455307893
document$.subscribe(function() {
  if (document.querySelector('.jp-Notebook')) {
    document.querySelector("div.md-sidebar.md-sidebar--primary").remove();
  }
});
