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

    // Send the user input to the backend for processing
    fetch('/chat', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({ message: userInput }),
    })
    .then(response => response.json())
    .then(data => {
        botSendMessage(data.response);
    })
    .catch(error => {
        console.error('Error:', error);
    });

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

    // Add the message bubble wrapped in a <p> tag
    let messageElement = document.createElement('p');
    messageElement.classList.add(className);
    messageElement.textContent = message;
    messageContainer.appendChild(messageElement);

    chatLog.appendChild(messageContainer);

    // Ensure the chat scrolls to the bottom after appending a new message
    setTimeout(() => {
        chatLog.scrollTop = chatLog.scrollHeight;
    }, 100); // Small delay to ensure the DOM is updated
}

function botSendMessage(message) {
    appendMessage(message, 'bot-message', 'Bot');
}

// Toggle Chat Functionality
document.getElementById('chat-header').addEventListener('click', function () {
    let chatContainer = document.getElementById('chat-container');
    let toggleIcon = document.getElementById('toggle-chat');

    chatContainer.classList.toggle('folded');
});
