// std
#include <iostream>
#include <exception>

// Boost
#include <boost/program_options.hpp>
#include <boost/filesystem.hpp>

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

int main(int argc, char* argv[])
{
	// Parse command line arguments
	string inputPath, outputPath, modelPath, deployPath, cfgPath;
    unsigned int verbose;
	try {
		options_description desc("Allowed options");
		desc.add_options()
			("help,h", "display the help message")
            ("verbose,v", value<unsigned int>(&verbose)->default_value(0), "output debug information")
			("input,i", value<string>(&inputPath)->required(), "image path")
			("output,o", value<string>(&outputPath)->required(), "output segmentation path")
            ("model,m", value<string>(&modelPath)->required(), "path to network weights model file  (.caffemodel)")
            ("deploy,d", value<string>(&deployPath)->required(), "path to network definition file for deployment (.prototxt)")
            ("cfg", value<string>(&cfgPath)->default_value("face_seg_image.cfg"), "configuration file (.cfg)")
			;
		variables_map vm;
		store(command_line_parser(argc, argv).options(desc).
			positional(positional_options_description().add("input", -1)).run(), vm);

        if (vm.count("help")) {
            cout << "Usage: face_seg_image [options]" << endl;
            cout << desc << endl;
            exit(0);
        }

        // Read config file
        std::ifstream ifs(vm["cfg"].as<string>());
        store(parse_config_file(ifs, desc), vm);

        notify(vm);

        if (!is_regular_file(inputPath)) throw error("input must be a path to an image!");
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
        // Initialize face segmentation
		face_seg::FaceSeg fs(deployPath, modelPath);
        
        // Read source image
        cv::Mat source_img = cv::imread(inputPath);

        // Do face segmentation
		cv::Mat seg = fs.process(source_img);
		if (seg.empty()) throw std::runtime_error("Face segmentation failed!");

        // Write output to file
        string filePath = outputPath;
        if (is_directory(outputPath))
        {
            path outputName = (path(inputPath).stem() += ".png");
            filePath = (path(outputPath) /= outputName).string();
        }
        cv::imwrite(filePath, seg); 
		
        // Debug
        if (verbose > 0)
        {
            // Write rendered image
			cv::Mat debug_render_img = source_img.clone();
			face_seg::renderSegmentationBlend(debug_render_img, seg);
            string debug_render_path = (path(filePath).parent_path() /=
                (path(filePath).stem() += "_debug.jpg")).string();
            cv::imwrite(debug_render_path, debug_render_img);
        }
		
	}
	catch (std::exception& e)
	{
		cerr << e.what() << endl;
		return 1;
	}

	return 0;
}

