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

    appendMessage(userInput, 'user-message', 'User');

    // Fake bot response for demonstration
    setTimeout(() => {
        let botResponse = "Hello! How can I assist you in finding the right product?";
        appendMessage(botResponse, 'bot-message', 'Bot');
    }, 1000);

    document.getElementById('user-input').value = '';
}

function appendMessage(message, className, sender) {
    let chatLog = document.getElementById('chat-log');

    // Create a container for the message and label
    let messageContainer = document.createElement('div');
    messageContainer.classList.add('message-container');

    // Add the user-container class for user messages
    if (sender === 'User') {
        messageContainer.classList.add('user-container');
    }

    // Add the label (user or bot)
    let label = document.createElement('span');
    label.classList.add('label');
    label.textContent = sender;
    messageContainer.appendChild(label);

    // Add the message bubble
    let messageElement = document.createElement('p');
    messageElement.classList.add(className);
    messageElement.textContent = message;
    messageContainer.appendChild(messageElement);

    chatLog.appendChild(messageContainer);
    chatLog.scrollTop = chatLog.scrollHeight;
}

// Toggle Chat Functionality (without the +/- icon)
document.getElementById('chat-header').addEventListener('click', function () {
    let chatContainer = document.getElementById('chat-container');
    chatContainer.classList.toggle('folded');
});
