import './App.css'
import member1 from './assets/team/member1.jpg'
import member2 from './assets/team/member2.jpg'
import member3 from './assets/team/member3.jpg'

const events = [
  {
    time: '05.0 — 08.0 s',
    title: 'Red Light',
    description: 'Traffic rule violation',
    type: 'normal',
  },
  {
    time: '20.0 — 26.0 s',
    title: 'Accident',
    description: 'Detected collision event',
    type: 'danger',
  },
  {
    time: '30.0 — 34.0 s',
    title: 'Jaywalking',
    description: 'Pedestrian traffic event',
    type: 'normal',
  },
  {
    time: '40.0 — 42.0 s',
    title: 'Near Miss',
    description: 'Potentially dangerous interaction',
    type: 'warning',
  },
  {
    time: '45.0 — 55.0 s',
    title: 'Wrong Way',
    description: 'Vehicle direction violation',
    type: 'normal',
  },
  {
    time: '20.0 — 35.0 s',
    title: 'Stopped Vehicle',
    description: 'Vehicle remaining stationary',
    type: 'normal',
  },
]

const scrollToSection = (id) => {
  document.getElementById(id)?.scrollIntoView({
    behavior: 'smooth',
  })
}

function App() {
  return (
    <div className="site">

      {/* NAVBAR */}
      <header className="navbar">
        <div className="nav-inner">
          <button
            className="brand"
            onClick={() => scrollToSection('home')}
          >
            <span className="brand-dot"></span>
            WIUT<span>CV</span>
          </button>

          <nav className="nav-links">
            <button onClick={() => scrollToSection('home')}>Home</button>
            <button onClick={() => scrollToSection('solution')}>Solution</button>
            <button onClick={() => scrollToSection('results')}>Results</button>
            <button onClick={() => scrollToSection('team')}>Team</button>
          </nav>

 <a
  href="https://github.com/Pac1FF1zM/traffic_checker"
  className="github-button"
  target="_blank"
  rel="noopener noreferrer"
>
  GitHub ↗
</a>
        </div>
      </header>


      {/* HERO */}
      <main>

        <section id="home" className="hero section">
          <div className="hero-grid">

            <div className="hero-content">
              <div className="eyebrow">
                WIUT HACKATHON 2026
              </div>

              <h1>
                Intelligent
                <span>Traffic Vision</span>
              </h1>

              <p className="hero-description">
                Computer vision system for detecting traffic events
                and anticipating dangerous situations from a fixed
                road camera.
              </p>

              <div className="hero-buttons">
                <button
                  className="primary-button"
                  onClick={() => scrollToSection('solution')}
                >
                  Explore solution
                  <span>→</span>
                </button>

                <button
                  className="secondary-button"
                  onClick={() => scrollToSection('results')}
                >
                  View results
                </button>
              </div>

              <div className="stats">
                <div className="stat">
                  <strong>2</strong>
                  <span>Task Parts</span>
                </div>

                <div className="stat">
                  <strong>6+</strong>
                  <span>Event Classes</span>
                </div>

                <div className="stat">
                  <strong>1</strong>
                  <span>Fixed Camera</span>
                </div>
              </div>
            </div>


            {/* CAMERA MOCKUP */}
            <div className="camera-wrapper">
              <div className="camera-card">

                <div className="camera-header">
                  <span>LIVE CAMERA</span>
                  <span className="live">
                    <i></i> LIVE
                  </span>
                </div>

                <div className="road">

                  <div className="road-line line-one"></div>
                  <div className="road-line line-two"></div>

                  <div className="car car-one">
                    <span></span>
                  </div>

                  <div className="car car-two">
                    <span></span>
                  </div>

                  <div className="car car-three">
                    <span></span>
                  </div>

                  <div className="detection detection-one">
                    <label>VEHICLE</label>
                  </div>

                  <div className="detection detection-risk">
                    <label>RISK</label>
                  </div>

                </div>

                <div className="camera-footer">
                  <span>Traffic analysis</span>
                  <strong>87% confidence</strong>
                </div>

              </div>
            </div>

          </div>
        </section>


        {/* SOLUTION */}
        <section id="solution" className="section light-section">
          <div className="section-heading">

            <div className="eyebrow">
              01 — SOLUTION
            </div>

            <h2>
              From video to
              <span> traffic intelligence.</span>
            </h2>

            <p>
              Our system processes a fixed CCTV stream and converts
              visual information into structured traffic events
              and accident-risk signals.
            </p>

          </div>


          <div className="feature-grid">

            <article className="feature-card">
              <div className="number">01</div>
              <div className="feature-icon">◉</div>

              <h3>Video Analysis</h3>

              <p>
                The system receives an MP4 video from the evaluation
                camera and processes its frames.
              </p>
            </article>


            <article className="feature-card">
              <div className="number">02</div>
              <div className="feature-icon">⌁</div>

              <h3>Event Detection</h3>

              <p>
                Detected traffic situations are converted into
                time segments with corresponding event labels.
              </p>
            </article>


            <article className="feature-card">
              <div className="number">03</div>
              <div className="feature-icon">△</div>

              <h3>Risk Estimation</h3>

              <p>
                The system produces a risk score over time using
                information available from previous frames.
              </p>
            </article>

          </div>
        </section>


        {/* ARCHITECTURE */}
        <section className="section architecture-section">
          <div className="section-heading">

            <div className="eyebrow">
              02 — ARCHITECTURE
            </div>

            <h2>
              How the pipeline
              <span> works.</span>
            </h2>

          </div>


          <div className="pipeline">

            <div className="pipeline-step">
              <div className="pipeline-number">01</div>

              <h3>Input Video</h3>

              <p>Fixed CCTV camera</p>
            </div>


            <div className="pipeline-line"></div>


            <div className="pipeline-step">
              <div className="pipeline-number">02</div>

              <h3>Vision Model</h3>

              <p>Frame-level analysis</p>
            </div>


            <div className="pipeline-line"></div>


            <div className="pipeline-step">
              <div className="pipeline-number">03</div>

              <h3>Tracking</h3>

              <p>Temporal information</p>
            </div>


            <div className="pipeline-line"></div>


            <div className="pipeline-step">
              <div className="pipeline-number">04</div>

              <h3>Events</h3>

              <p>Structured predictions</p>
            </div>

          </div>
        </section>


        {/* RESULTS */}
        <section id="results" className="section light-section">
          <div className="section-heading">

            <div className="eyebrow">
              03 — RESULTS
            </div>

            <h2>
              What the system
              <span> detects.</span>
            </h2>

            <p>
              The evaluation format represents traffic events
              as start time, end time and event class.
            </p>

          </div>


          <div className="results-grid">

            {events.map((event, index) => (
              <article
                className={`result-card ${event.type}`}
                key={index}
              >
                <div className="result-time">
                  {event.time}
                </div>

                <h3>{event.title}</h3>

                <p>{event.description}</p>
              </article>
            ))}

          </div>


          {/* RISK PANEL */}
          <div className="risk-panel">

            <div>
              <div className="eyebrow">
                ACCIDENT ANTICIPATION
              </div>

              <h3>
                Risk estimation over time
              </h3>

              <p>
                The system evaluates the possibility of an
                upcoming dangerous event using only previous frames.
              </p>
            </div>

            <div className="risk-score">
              <span>Current risk</span>
              <strong>87%</strong>
              <div className="risk-bar">
                <div></div>
              </div>
            </div>

          </div>

        </section>


        {/* TEAM */}
        <section id="team" className="section team-section">

          <div className="section-heading">

            <div className="eyebrow">
              04 — TEAM
            </div>

            <h2>
              The people behind
              <span> the system.</span>
            </h2>

            <p>
              Our team combines machine learning, system integration
              and product presentation.
            </p>

          </div>


<div className="team-grid">

  <article className="team-card">
    <div className="team-number">01</div>

    <img
      src={member1}
      className="team-photo"
      alt="ML Developer team member"
    />

    <h3>ML Developer</h3>

    <p>
      Computer vision models, training,
      inference and video analysis.
    </p>

    <span className="team-role">
      Machine Learning
    </span>
  </article>


  <article className="team-card">
    <div className="team-number">02</div>

    <img
      src={member2}
      className="team-photo"
      alt="System / Integration team member"
    />

    <h3>System / Integration</h3>

    <p>
      Repository integration, testing,
      inference, submission pipeline
      and documentation.
    </p>

    <span className="team-role">
      Engineering
    </span>
  </article>


  <article className="team-card">
    <div className="team-number">03</div>

    <img
      src={member3}
      className="team-photo"
      alt="Website & Presentation team member"
    />

    <h3>Website & Presentation</h3>

    <p>
      Project website, visual presentation,
      results dashboard and team documentation.
    </p>

    <span className="team-role">
      Product
    </span>
  </article>

</div>
          {/* FINAL CTA */}
          <div className="final-card">

            <div>
              <div className="eyebrow">
                WIUT HACKATHON 2026
              </div>

              <h2>
                Turning road video into
                <span> actionable intelligence.</span>
              </h2>

              <p>
                Computer Vision Track — Traffic Event Detection
                & Accident Anticipation
              </p>
            </div>

           <a
  href="https://github.com/Pac1FF1zM/traffic_checker"
  target="_blank"
  rel="noreferrer"
  className="primary-button final-button"
>
  View Repository →
</a>

          </div>

        </section>

      </main>


      {/* FOOTER */}
      <footer>
        <div>
          <strong>
            <span className="brand-dot"></span>
            WIUTCV
          </strong>
        </div>

        <p>
          WIUT Hackathon 2026 · Computer Vision Track
        </p>
      </footer>

    </div>
  )
}

export default App