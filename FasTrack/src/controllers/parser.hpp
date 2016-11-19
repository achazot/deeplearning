#ifndef PARSER_HPP
#define PARSER_HPP

#include <string>
#include <sstream>
#include <map>
#include <dirent.h>
#include <boost/tokenizer.hpp>

using namespace std;

class Parser
{

public:
	enum CommandType
	{
		CMD_LOADNET,
		CMD_QUIT,
		CMD_DETECT,
		CMD_AMBIGUOUS,
		CMD_UNKNOWN
	};

	Parser ( );
	int parseCommand ( string command );
	bool getLoadnetArgs ( string &caffemodel, string &prototxt, string &labels );
	bool getDetectArgs ( string &file );

private:
	map<int, string> m_aliases;
	string m_args;
};


#endif // PARSER_HPP
