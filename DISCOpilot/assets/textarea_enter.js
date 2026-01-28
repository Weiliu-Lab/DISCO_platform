(function () {
  if (window.__discoPromptEnterHandler) {
    return;
  }

  function handlePromptEnter(event) {
    if (!event || event.isComposing || event.keyCode === 229) {
      return;
    }
    var target = event.target;
    if (!target || target.id !== "prompt-input") {
      return;
    }
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      event.stopPropagation();
      var sendButton = document.getElementById("btn-send");
      if (sendButton) {
        sendButton.click();
      }
    }
  }

  window.__discoPromptEnterHandler = handlePromptEnter;
  document.addEventListener("keydown", handlePromptEnter, true);
})();
