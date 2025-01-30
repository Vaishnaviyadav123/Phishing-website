

const burger = document.querySelector('.burger');
const navLinks = document.querySelector('.nav-links');

burger.addEventListener('click', () => {
  navLinks.classList.toggle('active');
});


// script.js

document.getElementById('phishingForm').addEventListener('submit', function(event) {
  event.preventDefault();
  
  const url = document.getElementById('urlInput').value;
  const resultDiv = document.getElementById('result');
  
  // Simple phishing check (You can add more checks or API requests here)
  if (url.includes("phishing")) {
    resultDiv.innerHTML = "<span style='color: red;'>This URL is suspicious!</span>";
  } else {
    resultDiv.innerHTML = "<span style='color: green;'>This URL seems safe.</span>";
  }
});




document.getElementById("phishingForm").addEventListener("submit", async function(event) {
  event.preventDefault(); // Prevent the form from refreshing the page

  const urlInput = document.getElementById("urlInput").value; // Get the URL from the input field
  if (!urlInput) {
    alert("Please enter a URL.");
    return;
  }

  const resultElement = document.getElementById("result");
  resultElement.innerHTML = "Loading..."; // Display loading text while waiting for the response

  try {
    const response = await fetch("http://localhost:5000/predict", {
      method: "POST",
      headers: {
        "Content-Type": "application/json", // Inform backend you're sending JSON
      },
      body: JSON.stringify({ url: urlInput }), // Send the URL as a JSON object
    });

    // Check if the response is valid
    if (!response.ok) {
      throw new Error("Failed to fetch data from server");
    }

    const data = await response.json(); // Parse the JSON response

    if (data.result) {
      resultElement.innerHTML = `Result: <strong>${data.result}</strong>`;
    } else {
      resultElement.innerHTML = "Error: No result returned.";
    }
  } catch (error) {
    resultElement.innerHTML = `Error: ${error.message}`; // Display error message
    console.error(error);
  }
});
// Phishing Detection Logic (Basic Example)
document.getElementById('phishingForm').addEventListener('submit', function(event) {
  event.preventDefault();

  const urlInput = document.getElementById('urlInput').value;
  const resultDiv = document.getElementById('result');

  // Simulate a basic phishing check (just for demonstration)
  if (urlInput.includes("login") || urlInput.includes("account") || urlInput.includes("security") || urlInput.includes("verify")) {
    resultDiv.innerHTML = `<p style="color: red;">Warning! This URL seems suspicious.</p>`;
  } else {
    resultDiv.innerHTML = `<p style="color: green;">This URL seems safe!</p>`;
  }
});
