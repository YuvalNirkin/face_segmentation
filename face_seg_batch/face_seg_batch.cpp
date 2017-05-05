// std
#include <iostream>
#include <exception>

// Boost
#include <boost/program_options.hpp>
#include <boost/filesystem.hpp>
#include <boost/format.hpp>
#include <boost/regex.hpp>

// OpenCV
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

// face_seg
#include <face_seg/face_seg.h>
#include <face_seg/utilities.h>

using std::cout;
using std::endl;
using std::cerr;
using std::string;
using std::runtime_error;
using namespace boost::program_options;
using namespace boost::filesystem;

const std::string IMAGE_FILTER =
"(.*\\.(bmp|dib|jpeg|jpg|jpe|jp2|png|pbm|pgm|ppm|sr|ras))";

void getImagesFromDir(const std::string& dir_path, std::vector<std::string>& img_paths)
{
    boost::regex filter(IMAGE_FILTER);
    boost::smatch what;
    directory_iterator end_itr; // Default ctor yields past-the-end
    for (directory_iterator it(dir_path); it != end_itr; ++it)
    {
        // Skip if not a file
        if (!boost::filesystem::is_regular_file(it->status())) continue;

        // Get extension
        std::string ext = it->path().extension().string();
        std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);

        // Skip if no match
        if (!boost::regex_match(ext, what, filter)) continue;

        img_paths.push_back(it->path().string());
    }
}

void readImageListFromFile(const std::string& csv_file, std::vector<string>& img_paths)
{
    std::ifstream file(csv_file);
    string img_path;
    while (file.good())
    {
        std::getline(file, img_path, '\n');
        if (img_path.empty()) return;
        img_paths.push_back(img_path);
    }
}

void logError(std::ofstream& log, const string& img_path,
    const string& msg, bool write_to_file = true)
{
    std::cerr << "Error: " << msg << std::endl;
    if (write_to_file)
    {
        log << img_path << ",Error: " << msg << std::endl;
    }
}

int main(int argc, char* argv[])
{
	// Parse command line arguments
    string inputPath;
	string outputPath, modelPath, deployPath;
    string logPath, cfgPath;
    unsigned int verbose;
	try {
		options_description desc("Allowed options");
		desc.add_options()
			("help,h", "display the help message")
            ("verbose,v", value<unsigned int>(&verbose)->default_value(0), "output debug information")
            ("input,i", value<string>(&inputPath)->required(), "path to input directory or image list")
            ("output,o", value<string>(&outputPath)->required(), "output directory")
            ("model,m", value<string>(&modelPath)->required(), "path to network weights model file  (.caffemodel)")
            ("deploy,d", value<string>(&deployPath)->required(), "path to deploy prototxt file")
            ("log", value<string>(&logPath)->default_value("face_seg_batch_log.csv"), "log file path")
            ("cfg", value<string>(&cfgPath)->default_value("face_seg_batch.cfg"), "configuration file (.cfg)")
			;
		variables_map vm;
		store(command_line_parser(argc, argv).options(desc).
			positional(positional_options_description().add("input", -1)).run(), vm);

        if (vm.count("help")) {
            cout << "Usage: face_seg_batch [options]" << endl;
            cout << desc << endl;
            exit(0);
        }

        // Read config file
        std::ifstream ifs(vm["cfg"].as<string>());
        store(parse_config_file(ifs, desc), vm);

        notify(vm);

        if (!(is_regular_file(inputPath) || is_directory(inputPath)))
            throw error("input must be a path to input directory or image pairs list!");
        if (!is_directory(outputPath))
            throw error("output must be a path to a directory!");
        if (!is_regular_file(modelPath)) throw error("model must be a path to a file!");
        if (!is_regular_file(deployPath)) throw error("deploy must be a path to a file!");
	}
	catch (const error& e) {
        cerr << "Error while parsing command-line arguments: " << e.what() << endl;
        cerr << "Use --help to display a list of options." << endl;
		exit(1);
	}

	try
	{
        // Initialize log file
        std::ofstream log;
        if (verbose > 0)
            log.open(logPath);

        // Parse images
        std::vector<string> img_paths;
        if (is_directory(inputPath))
            getImagesFromDir(inputPath, img_paths);
        else readImageListFromFile(inputPath, img_paths);

		// Initialize face segmentation
		face_seg::FaceSeg fs(deployPath, modelPath);
        
        // For each image
        string prev_src_path, prev_tgt_path;
        cv::Mat source_img, target_img, rendered_img;
        for (const string& img_path : img_paths)
        {
            // Check if output image already exists
            path outputName = (path(img_path).stem() += ".png");
            string currOutputPath = (path(outputPath) /= outputName).string();
            if (is_regular_file(currOutputPath))
            {
                std::cout << "Skipping: " << outputName << std::endl;
                continue;
            }
            std::cout << "Face segmenting: " << outputName << std::endl;

			// Read source image
            cv::Mat source_img = cv::imread(img_path);

			// Do face segmentation
			cv::Mat seg = fs.process(source_img);
			if (seg.empty()) 
			{
				logError(log, img_path, "Face segmentation failed!", verbose);
				continue;
			}

            // Write output to file
            std::cout << "Writing " << outputName << " to output directory." << std::endl;
            cv::imwrite(currOutputPath, seg);

            // Debug
            if (verbose > 0)
            {
                // Write rendered image
				cv::Mat debug_render_img = source_img.clone();
				face_seg::renderSegmentationBlend(debug_render_img, seg);
                string debug_render_path = (path(outputPath) /=
                    (path(currOutputPath).stem() += "_debug.jpg")).string();
                cv::imwrite(debug_render_path, debug_render_img);
            }
        }
	}
	catch (std::exception& e)
	{
		cerr << e.what() << endl;
		return 1;
	}

	return 0;
}

