// script.js


document.getElementById('send-btn').addEventListener('click', sendMessage);
document.getElementById('user-input').addEventListener('keypress', function (e) {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault(); // Prevent new line
        sendMessage();
    }
});

let isPasswordMode = false;

function sendMessage() {
    let userInput = document.getElementById('user-input').value;
    if (userInput === '') return;

    // Mask the user input if in password mode
    let displayInput = isPasswordMode ? '*'.repeat(userInput.length) : userInput;

    appendMessage(displayInput, 'user-message', 'User');
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
        if (data.clear_chat) {
            clearChatLog(); // Clear chat if no past conversations
            if (data.session_date) {
                appendMessage(data.session_date, 'session-date', '');
            }
        }
        if (data.past_conversations && data.past_conversations.length > 0) {
            displayPastConversations(data.past_conversations);
            // Append the success message after displaying past conversations
            botSendMessage(data.response, data.session_date);
        } else {
            botSendMessage(data.response || data.message);  
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

    document.getElementById('user-input').value = '';
}

function clearChatLog() {
    let chatLog = document.getElementById('chat-log');
    chatLog.innerHTML = '';  // Clear the entire chat log
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
    return messageContainer;
}

function botSendMessage(message, sessionDate = null) {
    let messageElement;
    
    if (sessionDate) {
        appendMessage(sessionDate, 'session-date', '');  // Add date before welcome message
    }
    messageElement = appendMessage(message, 'bot-message', 'Flippey');

    // Scroll to the top of welcome message
    setTimeout(() => {
        if (message.includes("Welcome back!")) {
            console.log("Scrolling to Welcome Message:", messageElement); // Check if its scrolling to the correct position
            messageElement.scrollIntoView({ behavior: 'smooth', block: 'start' });
        }
    }, 300) // Allow the DOM to render fully
}

// Toggle Chat Functionality
document.getElementById('chat-header').addEventListener('click', function () {
    let chatContainer = document.getElementById('chat-container');

    chatContainer.classList.toggle('folded');
});

function displayPastConversations(conversations) {
    let chatLog = document.getElementById('chat-log');
    chatLog.innerHTML = ''; // Clear old messages

    let displayedSessions = new Set();

    conversations.forEach(chat => {
        if (!displayedSessions.has(chat.session_id)) {
            displayedSessions.add(chat.session_id);
            if (chat.session_date) {
                appendMessage(chat.session_date, 'session-date', '');
            }
        }
        if (chat.user) {
            appendMessage(chat.user, 'user-message', 'User');
        }
        if (chat.bot) {
            appendMessage(chat.bot, 'bot-message', 'Flippey');
        }
    });
}

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

function hideTypingIndicator() {
    let typingIndicator = document.getElementById('loading-indicator');
    if (typingIndicator) {
        clearInterval(typingIndicator.dataset.intervalId); // Stop animation
        typingIndicator.remove();
    }
}
