// script.js
document.getElementById('send-btn').addEventListener('click', sendMessage);
document.getElementById('user-input').addEventListener('keypress', function (e) {
    if (e.key === 'Enter') {
        sendMessage();
    }
});

function sendMessage() {
    let userInput = document.getElementById('user-input').value;
    if (userInput === '') return;

    appendMessage(userInput, 'user-message');

    // Fake bot response for demonstration
    setTimeout(() => {
        let botResponse = "Hello! How can I assist you in finding the right product?";
        appendMessage(botResponse, 'bot-message');
    }, 1000);

    document.getElementById('user-input').value = '';
}

function appendMessage(message, className) {
    let chatLog = document.getElementById('chat-log');
    let messageElement = document.createElement('p');
    messageElement.classList.add(className);
    messageElement.textContent = message;

    chatLog.appendChild(messageElement);
    chatLog.scrollTop = chatLog.scrollHeight;
}
