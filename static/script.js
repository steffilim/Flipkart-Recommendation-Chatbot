// script.js

// Event listener for the 'Send' button to trigger the sendMessage function
document.getElementById('send-btn').addEventListener('click', sendMessage);

// Event listener for the 'Enter' key in the input field to trigger the sendMessage function
document.getElementById('user-input').addEventListener('keypress', function (e) {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault(); // Prevent new line
        sendMessage();
    }
});

// Global flag to track whether the input field is in password mode (masked input)
let isPasswordMode = false;

/**
 * Sends the user's message to the backend and handles the response.
 * This function appends the user's message to the chat log, triggers the backend request,
 * and processes the response, including handling password mode and past conversations.
 */
function sendMessage() {
    let userInput = document.getElementById('user-input').value;
    if (userInput === '') return;

    // Mask the user input if in password mode
    let displayInput = isPasswordMode ? '*'.repeat(userInput.length) : userInput;

    // Append the user's message to the chat log
    appendMessage(displayInput, 'user-message', 'User');

    // Show typing indicator while waiting for the bot's response
    showTypingIndicator();

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
        hideTypingIndicator();

        // If the response contains clear_chat flag, clear the chat log
        if (data.clear_chat) {
            clearChatLog(); // Clear chat if no past conversations
        }

        // If the response includes past conversations, display them
        if (data.past_conversations && data.past_conversations.length > 0) {
            displayPastConversations(data.past_conversations);
            // Append the success message after displaying past conversations
            botSendMessage('You are now logged in! Do let me know what other product you are interested in! :)');
        } else if (data.clear_chat && data.message) {
            // For users with no past history, show the success message
            botSendMessage(data.message);
        } else {
            // Display the bot's response
            botSendMessage(data.response);
        }
        // Check if the response indicates that we are still in password mode
        if (data.response && (data.response.includes('Please enter your password.') || data.response.includes('Incorrect password.'))) {
            document.getElementById('user-input').type = 'password';  // Ensure input is masked
            isPasswordMode = true;  // Keep password mode active
        } else {
            document.getElementById('user-input').type = 'text';  // Unmask input field
            isPasswordMode = false;  // Reset password mode flag
        }
    })
    .catch(error => {
        hideTypingIndicator();
        console.error('Error:', error);
    });

    // Clear the input field after sending the message
    document.getElementById('user-input').value = '';
}

/**
 * Clears the chat log by removing all the messages in the chat container.
 */
function clearChatLog() {
    let chatLog = document.getElementById('chat-log');
    chatLog.innerHTML = '';  // Clear the entire chat log
}

/**
 * Appends a message to the chat log, either as a user or bot message.
 * It creates message containers and applies appropriate classes and labels.
 * @param {string} message - The message text to display.
 * @param {string} className - The CSS class to apply (e.g., 'user-message' or 'bot-message').
 * @param {string} sender - The sender label (e.g., 'User' or 'Flippey').
 */
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

/**
 * Appends a bot message to the chat log using a predefined class for bot messages.
 * @param {string} message - The message text to display.
 */
function botSendMessage(message) {
    appendMessage(message, 'bot-message', 'Flippey');
}

/**
 * Toggles the visibility of the chat container when the header is clicked.
 * This function folds or unfolds the chat window.
 */
document.getElementById('chat-header').addEventListener('click', function () {
    let chatContainer = document.getElementById('chat-container');

    chatContainer.classList.toggle('folded');
});

/**
 * Displays a list of past conversations in the chat log.
 * This will clear the existing chat log and append past user and bot messages.
 * @param {Array} conversations - An array of conversation objects containing user and bot messages.
 */
function displayPastConversations(conversations) {
    let chatLog = document.getElementById('chat-log');
    chatLog.innerHTML = ''; // Clear old messages

    conversations.forEach(chat => {
        if (chat.user) {
            appendMessage(chat.user, 'user-message', 'User');
        }
        if (chat.bot) {
            appendMessage(chat.bot, 'bot-message', 'Flippey');
        }
    });
}

/**
 * Shows a typing indicator to simulate the bot typing a response.
 * This includes an animated message that updates the "..." dots.
 */
function showTypingIndicator() {
    let typingIndicator = document.createElement('div');
    typingIndicator.id = 'loading-indicator';
    typingIndicator.classList.add('typing-indicator');
    typingIndicator.textContent = 'Flippey is typing';
    document.getElementById('chat-log').appendChild(typingIndicator);

    let dots = '';
    let dotCount = 0;
    let typingAnimation = setInterval(() => {
        dotCount = (dotCount + 1) % 4; 
        dots = '.'.repeat(dotCount);
        typingIndicator.textContent = `Flippey is typing${dots}`;
    }, 300);  

    typingIndicator.dataset.intervalId = typingAnimation;  
}

/**
 * Hides the typing indicator and stops its animation.
 */
function hideTypingIndicator() {
    let typingIndicator = document.getElementById('loading-indicator');
    if (typingIndicator) {
        clearInterval(typingIndicator.dataset.intervalId); // Stop animation
        typingIndicator.remove();
    }
}