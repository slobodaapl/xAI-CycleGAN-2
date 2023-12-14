$PythonScript = "./prepare_dataset.py"

# Additional arguments
$Arguments1 = "--start_index 0 --end_index 10 --stain he --type train"
$Arguments2 = "--start_index 10 --end_index 20 --stain he --type train"
$Arguments3 = "--start_index 20 --end_index 25 --stain he --type train"
$Arguments4 = "--start_index 0 --end_index 10 --stain p63 --type train"
$Arguments5 = "--start_index 10 --end_index 20 --stain p63 --type train"
$Arguments6 = "--start_index 20 --end_index 25 --stain p63 --type train"
$Arguments7 = "--start_index 0 --end_index 5 --stain he --type test"
$Arguments8 = "--start_index 0 --end_index 5 --stain p63 --type test"


# Start Python processes
Start-Process -FilePath "python" -ArgumentList "$PythonScript $Arguments1" -NoNewWindow -PassThru
Start-Process -FilePath "python" -ArgumentList "$PythonScript $Arguments2" -NoNewWindow -PassThru
Start-Process -FilePath "python" -ArgumentList "$PythonScript $Arguments3" -NoNewWindow -PassThru
Start-Process -FilePath "python" -ArgumentList "$PythonScript $Arguments4" -NoNewWindow -PassThru
Start-Process -FilePath "python" -ArgumentList "$PythonScript $Arguments5" -NoNewWindow -PassThru
Start-Process -FilePath "python" -ArgumentList "$PythonScript $Arguments6" -NoNewWindow -PassThru
Start-Process -FilePath "python" -ArgumentList "$PythonScript $Arguments7" -NoNewWindow -PassThru
Start-Process -FilePath "python" -ArgumentList "$PythonScript $Arguments8" -NoNewWindow -PassThru
