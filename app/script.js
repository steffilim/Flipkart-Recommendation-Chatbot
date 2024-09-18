// script.js

let loginStep = 0;  // Step to track login flow (0 - ask for userID, 1 - ask for password, 2 - logged in)
let userID = '';
let userPassword = '';
let isLoggedIn = false;

// Initialize the bot by asking for user ID
window.onload = function() {
    botSendMessage("Please enter your User ID to login:");
}

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
    
    if (!isLoggedIn) {
        handleLoginFlow(userInput);
    } else {
        // Normal conversation with the bot after login
        botSendMessage("Hello! How can I assist you in finding the right product?");
    }

    document.getElementById('user-input').value = '';
}

function handleLoginFlow(userInput) {
    if (loginStep === 0) {
        userID = userInput;
        botSendMessage("Please enter your password:");
        loginStep = 1;
    } else if (loginStep === 1) {
        userPassword = userInput;
        validateLogin(userID, userPassword);
    }
}

function validateLogin(userID, password) {
    // Simulate login validation
    if (userID === 'user123' && password === 'pass123') {
        isLoggedIn = true;
        botSendMessage("Login successful! How can I assist you today?");
    } else {
        botSendMessage("Invalid login credentials. Please enter your User ID to try again:");
        loginStep = 0; // Reset login process
    }
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
