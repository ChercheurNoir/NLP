
from pyngrok import ngrok
import subprocess

# Obtain the port of the Streamlit application
st_port = 8501  # You can use a different port if this one is already in use

# Define the command to run the Streamlit app
streamlit_command = f"streamlit run --server.port {st_port} streamlit_app.py"

# Launch the Streamlit app in a subprocess
streamlit_process = subprocess.Popen(streamlit_command, shell=True)

try:
    # Create an ngrok tunnel to make the app accessible via a public URL
    public_url = ngrok.connect(port=st_port)

    # Print the public URL
    print("Streamlit app is accessible at the URL:", public_url)

    # Keep the script running
    input("Press Enter to close the ngrok tunnel...\n")

finally:
    # Close the ngrok tunnel and the Streamlit process when the script is terminated
    ngrok.kill()
    streamlit_process.terminate()
