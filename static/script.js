// Handle image upload and display result
document.getElementById('uploadForm').addEventListener('submit', function(event) {
    event.preventDefault();  // Prevent the default form submission

    const formData = new FormData(this);
    const loader = document.getElementById('loader');
    const result = document.getElementById('result');
    const imageDescription = document.getElementById('imageDescription');
    const chatbotResponse = document.getElementById('chatbotResponse');

    // Hide previous results and show loader
    result.style.display = 'none';
    loader.style.display = 'block';

    fetch('/upload', {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        loader.style.display = 'none'; // Hide loader

        if (data.error) {
            alert('Error: ' + data.error);
        } else {
            imageDescription.textContent = data.description;
            chatbotResponse.textContent = data.response;
            result.style.display = 'block';
        }
    })
    .catch(error => {
        loader.style.display = 'none'; // Hide loader
        console.error('Error:', error);
        alert('An error occurred. Please try again.');
    });
});

// Handle chatbox interactions
document.getElementById('sendMessage').addEventListener('click', function() {
    const input = document.getElementById('chatInput');
    const message = input.value.trim();
    if (message === '') return;

    addMessageToChatbox('user', message);
    input.value = '';

    fetch('/chat', {
        method: 'POST',
        body: JSON.stringify({ message: message }),
        mode: 'cors',
        headers: {
            'Content-Type': 'application/json'
        }
    })
    .then(response => response.json())
    .then(data => {
        addMessageToChatbox('bot', data.response);
    })
    .catch(error => {
        console.error('Error:', error);
    });
});

document.getElementById('closeChatbox').addEventListener('click', function() {
    document.getElementById('chatbox').style.display = 'none';
});

function addMessageToChatbox(sender, message) {
    const chatboxContent = document.getElementById('chatboxContent');
    const messageDiv = document.createElement('div');
    messageDiv.classList.add('message');
    messageDiv.classList.add(`message-${sender}`);
    messageDiv.textContent = message;
    chatboxContent.appendChild(messageDiv);
    chatboxContent.scrollTop = chatboxContent.scrollHeight;  // Scroll to
}