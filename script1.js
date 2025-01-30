document.getElementById('phishingForm').addEventListener('submit', function (event) {
    event.preventDefault();
  
    const urlInput = document.getElementById('urlInput').value;
    const result = document.getElementById('result');
    const downloadPage = document.getElementById('downloadPage');
    const urlDetails = document.getElementById('urlDetails');
    const urlLocation = document.getElementById('urlLocation');
  
    // Fake phishing detection logic (replace with actual check)
    const isPhishing = Math.random() > 0.5; // Randomly simulate phishing
  
    if (isPhishing) {
      // Display "Not Safe" message and redirect to download page
      result.innerHTML = `<span style="color: red; font-weight: bold;">This URL is NOT SAFE!</span>`;
      
      // Display download page with URL details
      urlDetails.textContent = urlInput;
      urlLocation.textContent = 'Location: Fake Location (Simulated)';
      downloadPage.style.display = 'block';
    } else {
      // If URL is safe
      result.innerHTML = `<span style="color: green; font-weight: bold;">This URL is SAFE!</span>`;
    }
  });
  
  document.getElementById('downloadBtn').addEventListener('click', function () {
    // Generate a fake report and trigger download
    const reportContent = `Phishing URL Report\n\nURL: ${document.getElementById('urlDetails').textContent}\nLocation: ${document.getElementById('urlLocation').textContent}\n\nThis is a simulated phishing detection report.`;
  
    const blob = new Blob([reportContent], { type: 'text/plain' });
    const link = document.createElement('a');
    link.href = URL.createObjectURL(blob);
    link.download = 'phishing_report.txt';
    link.click();
  });
  