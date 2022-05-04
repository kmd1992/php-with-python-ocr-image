<?php
class PythonRunner
{
    public $file_path = '';

    public $out_put = '';

    public function pythonGrabber($pythonFilePathLaravelFormat)
    {
        $base_path = $_SERVER['DOCUMENT_ROOT'].$_SERVER['REQUEST_URI'];
        $this->file_path = $base_path . $pythonFilePathLaravelFormat;
    }

    public function run()
    {
        $command = "python " . $this->file_path . " 2>&1";
        $pid = popen($command, "r");
        while (!feof($pid)) {
            $this->out_put .= fread($pid, 256);
            flush();
            ob_flush();
            usleep(100000);
        }
        pclose($pid);
        return $this->out_put;
    }
}


class RunRun extends PythonRunner
{
    public function thisFile($pythonFilePathLaravelFormat)
    {
        $this->pythonGrabber($pythonFilePathLaravelFormat);
        return $this->run();
    }
}

$pythonScript = new RunRun();
$output = $pythonScript->thisFile('img_tbl_to_excel.py');
echo "<pre>";
print_r($output);