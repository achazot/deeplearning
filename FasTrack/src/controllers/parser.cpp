#include "parser.hpp"

bool compare_nocase ( const std::string& first, const std::string& second )
{
  unsigned int i=0;
  while ( (i<first.length()) && (i<second.length()) )
  {
    if (tolower(first[i])<tolower(second[i])) return true;
    else if (tolower(first[i])>tolower(second[i])) return false;
    ++i;
  }
  return ( first.length() < second.length() );
}

Parser::Parser()
{
	m_aliases[CMD_LOADNET] = "loadnetwork lnetwork";
	m_aliases[CMD_QUIT] = "quit exit x";
	m_aliases[CMD_DETECT] = "detect dt";
	m_aliases[CMD_TRACK] = "track";
	m_aliases[CMD_AMBIGUOUS] = "";

}

int Parser::parseCommand ( string line )
{
	while (line.c_str()[0]==' ') line = line.substr(1, line.size());
	string command = line.find(" ") == string::npos ? line : line.substr(0,  line.find_first_of(" "));
	m_args = line.find(" ") == string::npos ? "" : line.substr(line.find_first_of(" ") + 1, line.size());

	int corres = 0;
	int cmdtype = CMD_UNKNOWN;

	for (int i=0; i<CMD_UNKNOWN; i++)
	{
		if ( m_aliases[i].find(command) != string::npos )
		{
			cmdtype = i;
			corres++;
		}
	}

	if (corres>1) cmdtype = CMD_AMBIGUOUS;
	return cmdtype;
}

bool Parser::getDetectArgs ( string &file )
{
	file = m_args;
	return true;
}

bool Parser::getTrackArgs ( list<string> &files )
{
	DIR *dir;
	struct dirent *ent;
	if ( (dir = opendir ( m_args.c_str() ) ) != NULL )
	{
	  while ( ( ent = readdir (dir) ) != NULL )
	  {
			string fname(ent->d_name);
			if ( fname.find( ".jpg" ) != string::npos )
				files.push_back( m_args + "/" + fname );
	  }
	  closedir (dir);
	}
	else
	{
	  return false;
	}

	files.sort( compare_nocase );

	return true;
}

bool Parser::getLoadnetArgs ( string &caffemodel, string &prototxt, string &labels )
{
	if (m_args.find(" ") != string::npos)
	{
		boost::char_separator<char> cs(" ");
		boost::tokenizer<boost::char_separator<char>> tokens(m_args, cs);
		for (const auto& t : tokens)
		{
			if (t.find(".caffemodel") != string::npos)
				caffemodel = t;
			if (t.find(".prototxt") != string::npos || t.find(".pt") != string::npos)
				prototxt = t;
			if (t.find(".txt") != string::npos)
				labels = t;
		}
	}
	else // check if user specified a valid directory
	{
		DIR *dir;
		struct dirent *ent;
		if ( (dir = opendir ( m_args.c_str() ) ) != NULL )
		{
			while ( ( ent = readdir (dir) ) != NULL )
			{
				string fname(ent->d_name);
				if ( fname.find( ".caffemodel" ) != string::npos )
					caffemodel = m_args +"/"+ fname;
				if ( fname.find( ".prototxt" ) != string::npos || fname.find(".pt") != string::npos )
					prototxt = m_args +"/"+ fname;
				if ( fname.find( ".txt" ) != string::npos )
					labels = m_args +"/"+ fname;
			}
		 	closedir (dir);
		}
	}

	if (caffemodel.empty() || prototxt.empty() || labels.empty())
		return false;

	return true;
}
