const Importance = () => {
  return (
    <>
      <section className="py-20 bg-white">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="grid grid-cols-1 gap-8 lg:grid-cols-2 items-center">
            <div>
              <h2 className="text-3xl font-extrabold text-gray-900 mb-6">
                Why It Matters
              </h2>
              <p className="text-lg text-gray-600">
                Strain analysis unlocks detailed insights into regional
                myocardial health using cine MRI and deep learning. By capturing
                subtle changes in heart function, our approach empowers
                clinicians with precise diagnostics and accelerates the
                translation of research into practical cardiac care.
              </p>
            </div>
            <div className="relative h-64 lg:h-96">
              <img
                src="src/assets/Cine-CMR-Img.png"
                alt="Strain Map Visualization"
                className="w-full h-full object-cover rounded-lg shadow-lg"
              />
            </div>
          </div>
        </div>
      </section>
    </>
  );
};

export default Importance;
