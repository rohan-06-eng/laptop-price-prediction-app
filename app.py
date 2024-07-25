from flask import Flask, request, render_template
import pickle
import numpy as np

app = Flask(__name__)

# Load the model and data
pipe = pickle.load(open('pipe.pkl', 'rb'))
df = pickle.load(open('df.pkl', 'rb'))

# Define dropdown options based on df values
companies = df['Company'].unique()
types = df['TypeName'].unique()
cpu_brands = df['Cpu brand'].unique()
gpu_brands = df['Gpu brand'].unique()
os_list = df['os'].unique()


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        try:
            company = request.form.get('company', '')
            type_ = request.form.get('type', '')
            ram = int(request.form.get('ram', 0))
            weight = float(request.form.get('weight', 0))
            touchscreen = 1 if request.form.get('touchscreen', 'No') == 'Yes' else 0
            ips = 1 if request.form.get('ips', 'No') == 'Yes' else 0
            screen_size = float(request.form.get('screen_size', 0))
            resolution = request.form.get('resolution', '1920x1080')
            cpu = request.form.get('cpu', '')
            hdd = int(request.form.get('hdd', 0))
            ssd = int(request.form.get('ssd', 0))
            gpu = request.form.get('gpu', '')
            os = request.form.get('os', '')

            # Process resolution to calculate ppi
            X_res, Y_res = map(int, resolution.split('x'))
            ppi = ((X_res ** 2) + (Y_res ** 2)) ** 0.5 / screen_size

            # Ensure input values are within known categories or use a default value
            if company not in companies:
                company = companies[0]  # Default to the first company

            if type_ not in types:
                type_ = types[0]  # Default to the first type

            if cpu not in cpu_brands:
                cpu = cpu_brands[0]  # Default to the first CPU brand

            if gpu not in gpu_brands:
                gpu = gpu_brands[0]  # Default to the first GPU brand

            if os not in os_list:
                os = os_list[0]  # Default to the first OS

            # Prepare the feature vector for prediction
            query = np.array([company, type_, ram, weight, touchscreen, ips, ppi, cpu, hdd, ssd, gpu, os])
            query = query.reshape(1, -1)

            # Make prediction
            predicted_price = np.exp(pipe.predict(query)[0])

            return render_template('index.html',
                                   prediction=f"₹{int(predicted_price):,}",
                                   companies=companies,
                                   types=types,
                                   cpu_brands=cpu_brands,
                                   gpu_brands=gpu_brands,
                                   os_list=os_list,
                                   selected_company=company,
                                   selected_type=type_,
                                   selected_ram=ram,
                                   selected_weight=weight,
                                   selected_touchscreen=request.form.get('touchscreen', 'No'),
                                   selected_ips=request.form.get('ips', 'No'),
                                   selected_screen_size=screen_size,
                                   selected_resolution=resolution,
                                   selected_cpu=cpu,
                                   selected_hdd=hdd,
                                   selected_ssd=ssd,
                                   selected_gpu=gpu,
                                   selected_os=os)
        except Exception as e:
            return f"Error: {str(e)}"

    return render_template('index.html',
                           companies=companies,
                           types=types,
                           cpu_brands=cpu_brands,
                           gpu_brands=gpu_brands,
                           os_list=os_list)


if __name__ == '__main__':
    app.run(debug=True)
