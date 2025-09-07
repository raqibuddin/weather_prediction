
function toggleInputs() {
    const option = document.querySelector('input[name="option"]:checked')?.value;
    const cityBox = document.getElementById("cityInputBox");
    const coordBox = document.getElementById("coordInputBox");

    // Hide both first
    cityBox.classList.add("hidden");
    coordBox.classList.add("hidden");

    // Show the right one
    if (option === "_city_") {
        cityBox.classList.remove("hidden");
    } else if (option === "_coords_") {
        coordBox.classList.remove("hidden");
    }
}


    document.getElementById("weatherForm").addEventListener("submit", function(event) {
    const option = document.querySelector('input[name="option"]:checked');
    if (!option) {
        alert("Please select City or Coordinates.");
        event.preventDefault();
        return;
    }

    if (option.value === "_city_") {
        const city = document.getElementById("cityInput").value.trim();
        if (city === "") {
            alert("Please enter a city name.");
            event.preventDefault();
        }
    } else if (option.value === "_coords_") {
        const lat = parseFloat(document.getElementById("latInput").value.trim());
        const lon = parseFloat(document.getElementById("lonInput").value.trim());

        
    }
});


// ✅ Run on page load (restores correct box if Django preselected radio)
window.addEventListener("load", toggleInputs);

// ✅ Attach event listeners for toggle on radio change
document.querySelectorAll('input[name="option"]').forEach(radio => {
    radio.addEventListener("change", toggleInputs);
});


document.addEventListener("DOMContentLoaded", function () {
    const errorDiv = document.getElementById("error-message");

    if (errorDiv) {
        let errorMsg = errorDiv.getAttribute("data-message");
        let iconType = "error";   // default
        let titleText = "Oops..."; // default

        if (errorMsg.includes("Invalid API key")) {
            iconType = "warning";   
            titleText = "Invalid API Key";
        } else if (errorMsg.includes("City not found")) {
            iconType = "info";      
            titleText = "City Not Found";
        } else if (errorMsg.includes("Unable to connect")) {
            iconType = "question";  
            titleText = "Connection Error";
        } else {
            iconType = "error";     
            titleText = "Unexpected Error";
        }

        Swal.fire({
            icon: iconType,
            title: titleText,
            text: errorMsg,
            confirmButtonText: "OK"
        });
    }
});

