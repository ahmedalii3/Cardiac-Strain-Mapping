import { Heart } from "lucide-react";
const Footer = () => {
  return ( 
    <>
    <footer className="bg-gray-900 text-white">
        <div className="max-w-7xl mx-auto py-12 px-4 sm:px-6 lg:px-8">
          <div className="flex flex-col md:flex-row justify-between items-center">
            <div className="flex items-center mb-4 md:mb-0">
              <Heart className="h-8 w-8 text-blue-500" />
              <span className="ml-2 text-xl font-bold">StrainMap</span>
            </div>
            <div className="flex space-x-6">
              <a href="#" className="text-gray-300 hover:text-white">Privacy Policy</a>
              <a href="#" className="text-gray-300 hover:text-white">Terms of Use</a>
              <a href="mailto:info@strainmappingproject.com" className="text-gray-300 hover:text-white">Contact Us</a>
            </div>
          </div>
        </div>
      </footer>
    </>
   );
}
 
export default Footer;