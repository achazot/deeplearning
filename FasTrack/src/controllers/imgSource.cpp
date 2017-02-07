#include "imgSource.hpp"


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


ImgSource::ImgSource()
{

}

ImgSource::ImgSource(int device)
{
	m_useCam = true;
	if(!m_cap.open(device))
	{
		cout << "Cannot open camera." << endl;
		m_error = true;
	}
	else m_error = false;
}

ImgSource::ImgSource(string path)
{
	m_useCam = false;
	m_error = false;

	DIR *dir;
	struct dirent *ent;
	if ( (dir = opendir ( path.c_str() ) ) != NULL )
	{
		while ( ( ent = readdir (dir) ) != NULL )
		{
			string fname(ent->d_name);
			if ( fname.find( ".jpg" ) != string::npos || fname.find( ".png" ) != string::npos )
				m_files.push_back( path + (path.at(path.size()-1) ==  '/' ? fname : '/' + fname) );
		}
		closedir (dir);
	}
	else m_error = true;

	m_files.sort( compare_nocase );

	if (m_files.empty()) m_error = true;

}

ImgSource::~ImgSource()
{
	if (m_useCam && !m_error) m_cap.release();
}

Mat ImgSource::operator>> (Mat &sink)
{
	if (m_error) return sink;

	if (m_useCam)
	{
		m_cap >> sink;
	}
	else
	{
		if (!m_files.empty())
		{
			sink = imread(m_files.front());
			m_files.pop_front();
		}
	}

	return sink;
}

bool ImgSource::empty()
{
	if (m_useCam) return m_error;
	else return m_files.empty();
}
