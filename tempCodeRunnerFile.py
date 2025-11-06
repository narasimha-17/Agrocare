@app.route("/pesticide", methods=["GET", "POST"])
def pesticide():
    """
    Handles the pest and crop selection form and provides pesticide recommendations.
    
    The code has been updated to use the correct case-sensitive column names:
    'PestName', 'HostCrop', 'RecommendedPesticide', 'ApplicationMethod'.
    """
    
    # 1. Get unique values using correct capitalization
    try:
        pests = sorted(df['PestName'].unique())
        hostcrops = sorted(df['HostCrop'].unique())
    except KeyError as e:
        # Emergency fallback or detailed logging if the df is not loaded correctly
        print(f"ERROR: Column not found. Check if the DataFrame 'df' is loaded correctly. Details: {e}")
        pests = ["Error loading data"]
        hostcrops = ["Error loading data"]

    prediction = None

    if request.method == "POST":
        selected_pest = request.form.get("pest")
        selected_crop = request.form.get("hostcrop")

        # 2. Filter the DataFrame using correct capitalization
        result = df[
            (df['PestName'] == selected_pest) &
            (df['HostCrop'] == selected_crop)
        ]

        if not result.empty:
            # 3. Extract results using correct capitalization
            prediction = {
                "pesticide": result.iloc[0]['RecommendedPesticide'],
                "method": result.iloc[0]['ApplicationMethod']
            }
        else:
            prediction = {"pesticide": "No recommendation found", "method": "-"}

    return render_template("pesticide.html", pests=pests, hostcrops=hostcrops, prediction=prediction)
