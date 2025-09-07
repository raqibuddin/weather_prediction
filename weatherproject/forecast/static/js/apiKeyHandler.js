document.addEventListener("DOMContentLoaded", function () {
    const errorDiv = document.getElementById("error-message");

    if (errorDiv) {
        let errorMsg = errorDiv.getAttribute("data-message");

        Swal.fire({
            icon: "error",
            title: "API Key Error",
            text: errorMsg,
            confirmButtonText: "OK"
        });
    }
});
