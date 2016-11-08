/*
 *	tools.hpp
 *	A collection of various simple tools
 *
 */
#ifndef TOOLS_HPP
#define TOOLS_HPP

#include <string>

namespace tools
{
	/*
	 * Caseless string comparison function for "list.sort()" method
	 */
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
}

#endif // TOOLS_HPP
