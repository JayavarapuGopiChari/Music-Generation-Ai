// Basic UX polish: prevent double-submits and show a "working" state
document.addEventListener("DOMContentLoaded", () => {
  document.querySelectorAll("form").forEach((form) => {
    form.addEventListener("submit", () => {
      const btn = form.querySelector("button[type='submit']");
      if (btn && !btn.disabled) {
        btn.dataset.originalText = btn.textContent;
        btn.textContent = "Working ...";
        btn.disabled = true;
      }
    });
  });
});
