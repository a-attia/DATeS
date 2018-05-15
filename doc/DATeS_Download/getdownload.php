<?php
require '/opt/cslsoftwarehost/obfuscated_dates.php';

ini_set('display_errors', 1);
ini_set('display_startup_errors', 1);
error_reporting(E_ALL);

function sanitize_input($data)
{
    $data = strip_tags($data);
    $data = trim($data);
    $data = stripslashes($data);
    $data = htmlspecialchars($data);
    return $data;
}

function verify_token_validity($token)
{
    global $dbfilename;
    try {
        $filename = 'sqlite:' . $dbfilename;
        $dbh  = new PDO($filename);
        $query = "SELECT COUNT(*) as NUM FROM TIMEDACCESS WHERE TOKEN='" . trim($token) . "' AND  datetime('now') < EXPIRATIONTIME;";
        $result = $dbh->query($query);
        $count = $result->fetchAll()[0]['NUM'];
        $dbh = null;
        return Array("result" => $count);
    } catch (PDOException $e) {
        die("Database Error: " . $e->getMessage());
    }   
}

function send_file()
{
    global $realFileName, $distrodir, $fakeFileName;

    $file = trim($distrodir) . '/' . trim($realFileName);
    $fp = fopen($file, 'rb');

    header("Content-Type: application/octet-stream");
    header("Content-Disposition: attachment; filename=".trim($fakeFileName));
    header("Content-Length: " . filesize($file));
    fpassthru($fp);
}

$tokenErr = $token = "";

if ($_SERVER["REQUEST_METHOD"] == "GET") {
    if (empty($_GET["token"])) {
        $tokenErr = "Token is required";
    } else {
        $token = sanitize_input($_GET["token"]);
        // check if token only contains letters and numbers
        if (!preg_match("/^[a-zA-Z0-9]*$/", $token)) {
            $tokenErr = "Only letters and numbers allowed";
        }
    }

    if (!empty($tokenErr))
    {
        echo("There was a problem with the information provided: " . $tokenErr);
        die();
    }

    $result_array = verify_token_validity($token);
    
    if($result_array['result'] == 1)
    {
        send_file();
    }
    else
    {
?>
<!DOCTYPE html PUBLIC "-//W3C//DTD XHTML 1.0 Transitional//EN"
          "http://www.w3.org/TR/xhtml1/DTD/xhtml1-transitional.dtd">
<html xmlns="http://www.w3.org/1999/xhtml">
  <head>
    <meta http-equiv="Content-Type" content="text/html; charset=utf-8"/>

    <title>Download &#8212; DATeS 0.0.1 documentation</title>

    <link rel="stylesheet" href="_static/classic.css" type="text/css"/>
    <link rel="stylesheet" href="_static/pygments.css" type="text/css"/>

    <script type="text/javascript">
      var DOCUMENTATION_OPTIONS = {
      URL_ROOT: './',
      VERSION: '0.0.1',
      COLLAPSE_INDEX: false,
      FILE_SUFFIX: '.html',
      HAS_SOURCE: true,
      SOURCELINK_SUFFIX: '.txt'
      };
    </script>
    <script type="text/javascript" src="_static/jquery.js"></script>
    <script type="text/javascript" src="_static/underscore.js"></script>
    <script type="text/javascript" src="_static/doctools.js"></script>
    <script type="text/javascript"
            src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML"></script>
    <link rel="index" title="Index" href="genindex.html"/>
    <link rel="search" title="Search" href="search.html"/>
    <link rel="next" title="Install and Use DATeS" href="Installation.html"/>
    <link rel="prev" title="Introduction" href="Intro.html"/>
  </head>
  <body role="document">
    <div class="related" role="navigation" aria-label="related navigation">
      <h3>Navigation</h3>
      <ul>
        <li class="right" style="margin-right: 10px">
          <a href="genindex.html" title="General Index"
             accesskey="I">index</a></li>
        <li class="right">
          <a href="py-modindex.html" title="Python Module Index"
             >modules</a> |
        </li>
        <li class="right">
          <a href="Installation.html" title="Install and Use DATeS"
             accesskey="N">next</a> |
        </li>
        <li class="right">
          <a href="Intro.html" title="Introduction"
             accesskey="P">previous</a> |
        </li>
        <li class="nav-item nav-item-0"><a href="index.html">DATeS</a> &#187;</li>
      </ul>
    </div>

    <div class="document">
      <div class="documentwrapper">
        <div class="bodywrapper">
          <div class="body" role="main">
            <div class="section" id="download">
              <h1>Download<a class="headerlink" href="#download" title="Permalink to this headline">¶</a></h1>
              <embed>
                <style>
                  /* Full-width input fields */
                  input[type=text], input[type=password] {
                  width: 100%;
                  padding: 12px 20px;
                  margin: 8px 0;
                  display: inline-block;
                  border: 1px solid #ccc;
                  box-sizing: border-box;
                  }

                  /* Set a style for all buttons */
                  button {
                  background-color: #4CAF50;
                  color: white;
                  padding: 14px 20px;
                  margin: 8px 0;
                  border: none;
                  cursor: pointer;
                  width: 100%;
                  }

                  .center {
                  display: table;
                  margin: auto;
                  }

                  /* Extra styles for the cancel button */
                  .homebtn {
                  padding: 14px 20px;
                  background-color: #800000;
                  }

                  /* Extra styles for the cancel button */
                  .cancelbtn {
                  padding: 14px 20px;
                  background-color: #f44336;
                  }

                  /* Float cancel and signup buttons and add an equal width */
                  .cancelbtn, .signupbtn {
                  float: left;
                  width: 50%;
                  }

                  /* Float cancel and signup buttons and add an equal width */
                  .homebtn {
                  float: left;
                  width: 30%;
                  position: relative;
                  left: 35%;
                  }

                  /* Add padding to container elements */
                  .container {
                  padding: 16px;
                  }

                  /* Clear floats */
                  .clearfix::after {
                  content: "";
                  clear: both;
                  display: table;
                  }

                  /* Change styles for cancel button and signup button on extra small screens */
                  @media screen and (max-width: 300px) {
                  .cancelbtn, .signupbtn, .homebtn {
                  width: 100%;
                  }
                  }
                </style>
                <p>The download link is not valid. Please contact us on dates.assimilation@gmail.com to find out how we can help.</p>
              </embed>
            </div>
          </div>
        </div>
      </div>
      <div class="sphinxsidebar" role="navigation" aria-label="main navigation">
        <div class="sphinxsidebarwrapper">
          <p class="logo"><a href="index.html">
              <img class="logo" src="_static/DATeS_Logo.png" alt="Logo"/>
          </a></p>
          <h4>Previous topic</h4>
          <p class="topless"><a href="Intro.html"
                                title="previous chapter">Introduction</a></p>
          <h4>Next topic</h4>
          <p class="topless"><a href="Installation.html"
                                title="next chapter">Install and Use DATeS</a></p>
          <div id="searchbox" style="display: none" role="search">
            <h3>Quick search</h3>
            <form class="search" action="search.html" method="get">
              <div><input type="text" name="q"/></div>
              <div><input type="submit" value="Go"/></div>
              <input type="hidden" name="check_keywords" value="yes"/>
              <input type="hidden" name="area" value="default"/>
            </form>
          </div>
          <script type="text/javascript">$('#searchbox').show(0);</script>
        </div>
      </div>
      <div class="clearer"></div>
    </div>
    <div class="related" role="navigation" aria-label="related navigation">
      <h3>Navigation</h3>
      <ul>
        <li class="right" style="margin-right: 10px">
          <a href="genindex.html" title="General Index"
             >index</a></li>
        <li class="right">
          <a href="py-modindex.html" title="Python Module Index"
             >modules</a> |
        </li>
        <li class="right">
          <a href="Installation.html" title="Install and Use DATeS"
             >next</a> |
        </li>
        <li class="right">
          <a href="Intro.html" title="Introduction"
             >previous</a> |
        </li>
        <li class="nav-item nav-item-0"><a href="index.html">DATeS</a> &#187;</li>
      </ul>
    </div>
    <div class="footer" role="contentinfo">
      &#169; Copyright 2017, Ahmed Attia and adrian Sandu.
      Created using <a href="http://sphinx-doc.org/">Sphinx</a> 1.5.2.
    </div>
  </body>
</html>
<?php
    
    }
}
?>
